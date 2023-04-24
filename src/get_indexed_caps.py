import sys
import json
import os.path
import logging
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import clip
from collections import defaultdict
from PIL import Image
import faiss 
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True 
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


class ClipRetrieval():
    def __init__(self, index_name):
        self.datastore = faiss.read_index(index_name)
        self.datastore.nprobe=25
    
    def get_nns(self, query_img, k=20):
        #get k nearest image
        D, I = self.datastore.search(query_img, k)
        return D, I[:,:k]


class EvalDataset():

    def __init__(self, dataset_splits, images_dir,  images_names, clip_retrieval_processor, eval_split="val_images"):
        super().__init__()

        with open(dataset_splits) as f:
            self.split = json.load(f)

        self.split = self.split[eval_split]
        self.images_dir= images_dir

        with open(args.images_names) as f:
            self.images_names = json.load(f)

        self.clip_retrieval_processor = clip_retrieval_processor

    def __getitem__(self, i):
        coco_id = self.split[i]

        image_filename= self.images_dir+self.images_names[coco_id]
        img_open = Image.open(image_filename).copy()
        img = np.array(img_open)
        if len(img.shape) ==2 or img.shape[-1]!=3: #convert grey or CMYK to RGB
            img_open = img_open.convert('RGB')
    
        inputs_features_retrieval = self.clip_retrieval_processor(img_open).unsqueeze(0)
        return inputs_features_retrieval.detach(), coco_id

    def __len__(self):
        return len(self.split)


def evaluate(args):

    #load data of the datastore (i.e., captions)
    with open(args.index_captions) as f:
        data_datastore = json.load(f)

    print("loading datastore")
    datastore = ClipRetrieval(args.datastore_path)
    datastore_name = args.datastore_path.split("/")[-1]
    
    #load clip to encode the images that we want to retrieve captions for
    clip_retrieval_model, clip_retrieval_feature_extractor = clip.load("RN50x64", device=device)
    
    #data_loader to get images that we want to retrieve captions for
    data_loader = torch.utils.data.DataLoader(
        EvalDataset(
            args.dataset_splits, 
            args.images_dir,
            args.images_names,
            clip_retrieval_feature_extractor, 
            args.split), 
        batch_size=1, 
        shuffle=True,  
        num_workers=0, 
        pin_memory=True
    )

    nearest_caps={}
    with torch.no_grad():
        for data in tqdm(data_loader):
                
            inputs_features_retrieval, coco_id =data
            coco_id = coco_id[0]

            #normalize images to retrieve (since datastore has also normalized captions)
            inputs_features_retrieval = inputs_features_retrieval.to(device)
            image_retrieval_features = clip_retrieval_model.encode_image(inputs_features_retrieval[0])  
            image_retrieval_features /= image_retrieval_features.norm(dim=-1, keepdim=True)
            
        D, nearest_ids=datastore.get_nns(image_retrieval_features.detach().cpu().numpy().astype(np.float32), k=10)
        
        #Since at inference batch is 1
        D=D[0]
        nearest_ids=nearest_ids[0]

        list_of_similar_caps=defaultdict(list)
        for index in range(len(nearest_ids)):
            nearest_id = str(nearest_ids[index])            
            nearest_cap=data_datastore[nearest_id]
    
            if len(nearest_cap.split()) > args.max_caption_len:
                print("retrieve cap too big" )
                continue

            distance=D[index]
            list_of_similar_caps[datastore_name].append((nearest_cap, str(distance)))

    nearest_caps[str(coco_id)]=list_of_similar_caps


    #save results
    outputs_dir = os.path.join(args.output_path, "retrieved_caps")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    data_name=args.dataset_splits.split("/")[-1]

    name = "nearest_caps_"+data_name +"_w_"+datastore_name + "_"+ args.split
    results_output_file_name = os.path.join(outputs_dir, name + ".json")
    json.dump(nearest_caps, open(results_output_file_name, "w"))



def check_args(args):
    parser = argparse.ArgumentParser()
    
    #Info of the dataset to evaluate on (vizwiz, flick30k, msr-vtt)
    parser.add_argument("--images_dir",help="Folder where the preprocessed image data is located", default="data/vizwiz/images")
    parser.add_argument("--dataset_splits",help="File containing the dataset splits", default="data/vizwiz/dataset_splits.json")
    parser.add_argument("--images_names",help="File containing the images names per id", default="data/vizwiz/images_names.json")
    parser.add_argument("--split", default="val_images", choices=["val_images", "test_images"])
    parser.add_argument("--max-caption-len", type=int, default=25)

    #Which datastore to use (web, human)
    parser.add_argument("--datastore_path", type=str, default="datastore/vizwiz/vizwiz")
    parser.add_argument("--index_captions",
                        help="File containing the captions of the datastore per id", default="datastore/vizwiz/vizwiz.json")
    parser.add_argument("--output-path",help="Folder where to store outputs", default="eval_vizwiz_with_datastore_from_vizwiz.json")          

    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info(args)
    evaluate(args)


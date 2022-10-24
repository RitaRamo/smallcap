import os
import sys
import pandas as pd
import json
from tqdm import tqdm 
from PIL import Image
import torch
from multiprocessing import Pool
import h5py
from transformers import logging
from transformers import CLIPFeatureExtractor, CLIPVisionModel
logging.set_verbosity_error()

data_dir = 'data/images/'
features_dir = 'features/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_name = 'openai/clip-vit-base-patch32'
feature_extractor = CLIPFeatureExtractor.from_pretrained(encoder_name) 
clip_encoder = CLIPVisionModel.from_pretrained(encoder_name).to(device)

annotations = json.load(open('data/dataset_coco.json'))['images']

def load_data():
    data = {'train': [], 'val': []}

    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if item['split'] == 'train' or item['split'] == 'restval':
            data['train'].append({'file_name': file_name, 'cocoid': item['cocoid']})
        elif item['split'] == 'val':
            data['val'].append({'file_name': file_name, 'cocoid': item['cocoid']})
    return data

def encode_split(data, split):
    df = pd.DataFrame(data[split])

    bs = 256
    h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')
    for idx in tqdm(range(0, len(df), bs)):
        cocoids = df['cocoid'][idx:idx + bs]
        file_names = df['file_name'][idx:idx + bs]
        images = [Image.open(data_dir + file_name).convert("RGB") for file_name in file_names]
        with torch.no_grad(): 
            pixel_values = feature_extractor(images, return_tensors='pt').pixel_values.to(device)
            encodings = clip_encoder(pixel_values=pixel_values).last_hidden_state.cpu().numpy() 
        for cocoid, encoding in zip(cocoids, encodings):
            h5py_file.create_dataset(str(cocoid), (50, 768), data=encoding)

data = load_data()

encode_split(data, 'train')
encode_split(data, 'val')

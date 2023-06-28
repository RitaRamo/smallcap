# SmallCap

> We now have a demo, check it out: https://huggingface.co/spaces/RitaParadaRamos/SmallCapDemo :v:

## Dependencies

The code was developed in Python 3.9.

```
conda create -n smallcap python=3.9
conda activate smallcap
pip install -r requirements.txt
```

#### Evaluation package

Download Stanford models for computing SPICE (a slightly modified version of this [repo](https://github.com/daqingliu/coco-caption.git)):

```./coco-caption/get_stanford_models.sh```

## Interacting with SmallCap

Our pretrained model is available on HuggingFace at `Yova/SmallCap7M`. 

To use it, you also need the retrieval datastore:

```
mkdir datastore
```

Download the COCO [index](https://drive.google.com/file/d/1ZP5I-xbjaNU7cU48C_ctHd95SaA0jBHe/view?usp=sharing) and associated [captions](https://drive.google.com/file/d/1BT0Qc6g40fvtnJ_yY0aipfCuCMgu5qaR/view?usp=sharing) and place them in `datastore/`.

See `SmallCap_demo.inynb` for a demo of our pretrained model.

## Training SmallCap

<details>
<summary>Click to expand</summary>

### Data

Download the COCO Karpathy splits file `dataset_coco.json` from [here](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits) and place it in `data/`.

Download all COCO images (train, val and test, 2017 version) from [here](https://cocodataset.org/#download) and place them in `data/images`. The expected naming format is twelve digits followed by a `.jpg` extension, e.g. `data/images/000000000001.jpg` for image with COCO id `1`.

### Preprocessing

At the moment CLIP models based on ResNet are not available through HuggingFace so it is necessary to also install the original CLIP implementation from [here](https://github.com/openai/CLIP):

```
pip install git+https://github.com/openai/CLIP.git
```

Extract train and val features: 

```
mkdir features
python src/extract_features.py
```

Retrieve captions

```python src/retrieve_captions.py```

### Model training

```python train.py```

Models are saved under name <rag/norag>_<num params>M, e.g. `rag_7M` for a model trained with retrieval augmentation and 7M trainable parameters.

### Inference

```python infer.py --model_path <MODEL_PATH>```

If you also specify `--checkpoint_path` inference runs with only that checkpoint. Else, all checkpoints in `--model_path` are used. 

If you specify `--infer_test` inference uses test data, else val data is used.

E.g. to run inference on the test split with model `rag_7M`, checkpoint `17712`, run

```python infer.py --model_path experiments/rag_7M --checkpoint_path checkpoint-17712 --infer_test```

The model predictions are stored as ```<val/test>_preds.json``` in each respective checkpoint subdirectory.

Note: You can safely ignore the warning `Some weights of ThisGPT2LMHeadModel were not initialized from the model checkpoint at gpt2 and are newly initialized...` It occurs because a new model is first built and then the pre-trained parameters are loaded into it. 

### Evaluate predictions

```python coco-caption/run_eval.py <GOLD_ANN_PATH> <PREDICTIONS_PATH>```
</details>



### Paper

If you find our code/data/models or ideas useful in your research, please consider citing the [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Ramos_SmallCap_Lightweight_Image_Captioning_Prompted_With_Retrieval_Augmentation_CVPR_2023_paper.pdf):
```
@article{ramos2022smallcap,
  title={SmallCap: Lightweight Image Captioning Prompted with Retrieval Augmentation},
  author={Ramos, Rita and Martins, Bruno and Elliott, Desmond and Kementchedjhieva, Yova},
  journal={CVPR},
  url={https://openaccess.thecvf.com/content/CVPR2023/papers/Ramos_SmallCap_Lightweight_Image_Captioning_Prompted_With_Retrieval_Augmentation_CVPR_2023_paper.pdf},
  year={2023}
}
```














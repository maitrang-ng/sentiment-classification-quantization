# ProtonX MLE 


<p align="center">
    <img src='https://storage.googleapis.com/protonx-cloud-storage/transformer/protonx-transf.png' width=200 class="center">
</p>

Project: Fine-tune & quantize Distilbert for IMDB reviews classification.
Docker, Flask, ONNX, ONNXRuntime, Transformers, python 3.9


Authors:
- Github: maitrang-ng, NamTran072, quangle, mp1704
- Email: maitrang2406@gmail.com, lenhatquang1991@gmail.com, tranhainam072@gmail.com, pmphuong1704@gmail.com

Advisors:
- Github: bangoc123, 
- Email: ngoc@protonx.io

## I.  Set up environment locally
- Step 1: 
Python 3.9 & pipenv are required

- Step 2: 
```cd``` into sentiment-classification then use command line ```pipenv install -r requirements.txt``` to set up env
```pipenv check``` to check env status
```pipenv shell``` to active env

- Step 3: 
```cd src ```in command line


## II. Set up docker

```
docker build -t sentiment-classification .
docker run -dp 5001:80 sentiment-classification
```

## III. Set up your dataset

- Datasets must be in format CSV. You can check dataset folder to more details.

## IV. Training Process

Training script:
```
python3 trainer.py -ds ${dataset-path} -r ${ratio-split-dataset} -m ${base-model} -b ${batcg-size}
 -e ${epoches} -lr ${learning-rate} -wd ${weight-decay} -o ${output-path} -s ${save-model-dir}
```
Example: you can ignore all arguments then command will use the default setting
```
python3 trainer.py -s ../models/DistilBert
``` 

dataset-path can accept 1 or more files path, then split dataset with the ratio as you choose.
If you provide 2 or 3 files path and do not provide split ratio, the files will be considered in order: train, test or train, valid, test.
For 1 or more than 3 files path, split ration is required.
Split ratio can accept 1 or 2 values. One value: test-dataset ratio. Two ratios in order: test-dataset ration, validation-dataset ratio.
Example: 
```
python3 trainer.py -ds ../dataset/train.csv -r 0.2 0.2 -s ../models/DistilBert
``` 

There are some important arguments for the script you should consider when running it:
- `-s`: the directory to save model

## V. Predict Process
Predict with torch model:
```
python3 predictor.py torch -m ${torch-model-path}
```
Example:
```
python3 predictor.py -m ../models/DistilBert
``` 

Predict with onnx model:
```
python3 predictor.py onnx -m ${onnx-model-path} -t ${tokenizer-path}
```
Example:
```
python3 predictor.py -m ../models/ONNX/distilbert_full_precision.onnx -t ../models/ONNX
``` 


## VI. Evaluation
Run locally on 1000 examples of test dataset

Evaluation torch model

```
python3 evaluation.py torch

```
Result:
```
test Acc: 0.9355
PyTorch cpu Inference time = 323.8157 ms
```


Evaluation ONNX model

```
python3 evaluation.py onnx
```
Result:
```
test Acc: 0.9482
ONNX cpu Inference time = 286.9295 ms
```





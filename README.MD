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

## I.  Set up environment
- Step 1: 

```
Python 3.9 & pipenv are required
```

- Step 2: 
```
cd into sentiment-classification then use command line pipenv install -r requirements.txt to set up env
pipenv check to check env status
pipenv shell to active env
```

- Step 3: 

```
cd src in command line

``` 

## II.  Set up your dataset

- Datasets must be in format CSV. You can check dataset folder to more details.
- References: [NLP](https://github.com/bangoc123/transformer) and [CV](https://github.com/bangoc123/mlp-mixer)

## III. Training Process

Training script:

```python

python3 trainer.py -s ${save-model-dir}

```

Example:

```python

python3 train.py -s '../models/DistilBert'

``` 

There are some important arguments for the script you should consider when running it:

- `train-folder`: The folder of training data
- `valid-folder`: The folder of validation data
- ...

## IV. Predict Process

```bash
python predict.py --test-data ${link_to_test_data}
```

## V. Result and Comparision

**FIXME**

Your implementation
```
Epoch 7/10
782/782 [==============================] - 261s 334ms/step - loss: 0.8315 - acc: 0.8565 - val_loss: 0.8357 - val_acc: 0.7978
Epoch 8/10
782/782 [==============================] - 261s 334ms/step - loss: 0.3182 - acc: 0.8930 - val_loss: 0.6161 - val_acc: 0.8047
Epoch 9/10
782/782 [==============================] - 261s 333ms/step - loss: 1.1965 - acc: 0.8946 - val_loss: 3.9842 - val_acc: 0.7855
Epoch 10/10
782/782 [==============================] - 261s 333ms/step - loss: 0.4717 - acc: 0.8878 - val_loss: 0.4894 - val_acc: 0.8262

```

**FIXME**

Other architecture

```
Epoch 6/10
391/391 [==============================] - 115s 292ms/step - loss: 0.1999 - acc: 0.9277 - val_loss: 0.4719 - val_acc: 0.8130
Epoch 7/10
391/391 [==============================] - 114s 291ms/step - loss: 0.1526 - acc: 0.9494 - val_loss: 0.5224 - val_acc: 0.8318
Epoch 8/10
391/391 [==============================] - 115s 293ms/step - loss: 0.1441 - acc: 0.9513 - val_loss: 0.5811 - val_acc: 0.7875
```

Your comments about these results <--- **FIXME**


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```



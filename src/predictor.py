import os
import torch
import onnxruntime as ort
import numpy as np

from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from constant import torch_model_path, onnx_model_path, onnx_path, max_seq_length
from utils import softmax

def torch_predict(model_path, input):
  # Load model & tokenizer
  model = AutoModelForSequenceClassification.from_pretrained(model_path)
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  classes = ['Negative', 'Positive']
  # Turn on inference context manager
  with torch.inference_mode():
    tokenized_input = tokenizer(input, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
    # Forward pass
    logit = model(**tokenized_input).logits
    # Predict result
    result = {
        'label': classes[logit[0].argmax(-1)],
        'score': softmax(logit)[0].max().item()
        }
  return result

def onnx_predict(model_path, tokenizer_path, input):
  # Load onnx model & tokenizer
  ort_sess = ort.InferenceSession(model_path)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
  classes = ['Negative', 'Positive']
  tokenized_input = tokenizer(input, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='np')
  output = ort_sess.run(None, {'input_ids': tokenized_input['input_ids'],
                              'attention_mask': tokenized_input['attention_mask']})
  # Predict result
  result = {
      'label': classes[output[0][0].argmax(-1)],
      'score': softmax(output[0])[0].max()
      }
  return result

def cli():
  parser = ArgumentParser(prog = 'predict.py',
                          description='Sentiment prediction for film review')
  subparser = parser.add_subparsers(dest='command')
  parser_torch = subparser.add_parser("torch", help= "Inference with torch model.")
  parser_onnx = subparser.add_parser("onnx", help="Inference with onnx model.")

  parser_torch.add_argument("--model-path",
                            "-m",
                            #required=True,
                            default=torch_model_path,
                            type=str,
                            help ="Predict with torch model.")
  
  parser_onnx.add_argument("--model-path",
                            "-m",
                            #required=True,
                            default=onnx_model_path,
                            type=str,
                            help ="Predict with torch model.")
  parser_onnx.add_argument("--tokenizer-path",
                           "-t",
                           #required=True,
                           default=onnx_path, 
                           type=str,
                           help ="Predict with onnx model.")

  args = parser.parse_args()
  return args


if __name__ == "__main__":
  args = cli()
  text = input("Input text to analyze sentiment: ")
  if not bool(text.strip()):
    print('Text must be not empty.')
  else:
    if args.command == 'torch':
      result = torch_predict(args.model_path, text)
    elif args.command == 'onnx':
      result = onnx_predict(args.model_path, args.tokenizer_path, text)
    else:
      result = "ERROR: You must choose torch or onnx to predict. Example: python3 predictor.py torch"
    print(result)
  






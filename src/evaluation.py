import torch
import time
import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from constant import torch_model_path, onnx_model_path
from datasets import load_dataset
from argparse import ArgumentParser

def evaluate_torch(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    model.to(device)

    # Put model in eval mode
    model.eval()
    max_seq_length = model.config.max_position_embeddings

    # Setup test loss and test accuracy values
    test_acc=0

    # Turn on inference context manager
    with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, X in enumerate(dataloader):
            # Send data to target device
            inputs = {'input_ids': X['input_ids'].to(device),
                      'attention_mask': X['attention_mask'].to(device)}
                      #.reshape(1, max_seq_length)}''
            outputs = X['label'].to(device)
            # Forward pass
            logits = model(**inputs).logits

            # Calculate and accumulate accuracy
            test_pred_labels = logits.argmax(dim=1)
            #test_acc += (test_pred_labels == outputs)
            test_acc += ((test_pred_labels == outputs).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average accuracy
    test_acc = test_acc / len(dataloader)
    return  test_acc

# Measure the latency. It is not accurate using Jupyter Notebook, it is recommended to use standalone python script.
def time_evaluation_torch(model, dataloader, loss_fn, device):
  eval_start_time = time.time()
  result = evaluate_torch(model, dataloader, loss_fn, device)
  eval_end_time = time.time()
  eval_duration_time = eval_end_time - eval_start_time
  print(f'test Acc: {result:.4f}')
  print("PyTorch {} Inference time = {} ms".format(device.type, format(eval_duration_time * 1000 / len(dataloader.dataset), '.4f')))


def evaluate_onnx(onnx_model_path, dataloader, device):
  ort_sess = ort.InferenceSession(onnx_model_path)
  test_acc=0
  # Loop through DataLoader batches
  for batch, X in enumerate(dataloader):
    # Send data to target device
    inputs = {'input_ids': X['input_ids'].cpu().detach().numpy(),
              'attention_mask': X['attention_mask'].cpu().detach().numpy()}
    outputs = X['label'].cpu().detach().numpy()
    # Forward pass
    logits = np.reshape(ort_sess.run(None, inputs), (-1,2))
    # Calculate and accumulate accuracy
    test_pred_labels = logits.argmax(-1)
    test_acc += ((test_pred_labels == outputs).sum()/len(test_pred_labels))
      # Adjust metrics to get average accuracy
  test_acc = test_acc / len(dataloader)
  return  test_acc

def time_evaluation_onnx(onnx_model_path, dataloader, device):
  eval_start_time = time.time()
  result = evaluate_onnx(onnx_model_path, dataloader, device)
  eval_end_time = time.time()
  eval_duration_time = eval_end_time - eval_start_time
  print(f'test Acc: {result:.4f}')
  print("ONNX {} Inference time = {} ms".format(device, format(eval_duration_time * 1000 / len(dataloader.dataset), '.4f')))

# Load test dataset
imdb = load_dataset("imdb", split="train+test")
split_dataset = imdb.train_test_split(test_size=0.02, stratify_by_column='label', shuffle=True, seed=42)
test_dataset = split_dataset['test']

if __name__ =='__main__':
    # Define loss
    loss_fn = nn.CrossEntropyLoss()
    # Define model & tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(torch_model_path)
    tokenizer = AutoTokenizer.from_pretrained(torch_model_path)
    # Define test dataloader
    test_dataset = test_dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=512), batched=True)
    test_dataset.set_format(type='torch', columns=['label', 'input_ids', 'attention_mask'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    device = torch.device("cpu")

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser_torch = subparsers.add_parser("torch")
    parser_onnx = subparsers.add_parser("onnx")
    args = parser.parse_args()

    if args.command=='torch':
        time_evaluation_torch(model=model,
                            dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            device=device)
    elif args.command=='onnx':
       time_evaluation_onnx(onnx_model_path=onnx_model_path, dataloader=test_dataloader,device=device)
    else:
       print('ERROR: Choose onnx model or torch model to make evaluation.')

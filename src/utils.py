import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
  labels = eval_pred.label_ids
  preds = eval_pred.predictions.argmax(-1)
  f1 = f1_score(labels, preds, average="weighted")
  acc = accuracy_score(labels, preds, normalize=True)
  return {"accuracy": acc, "f1 score": f1}

'''def accuracy_score(y_true, y_pred, normalize=True):
  score=0
  if normalize==True:
    try: 
        score += (y_pred==y_true).mean()
    except:
       print('Check format & dimention of inputs')
       score = "error"
  elif normalize==False:
    try: 
        score += (y_pred==y_true).sum()
    except:
       print('Check format & dimention of inputs')
       score = "error"
  return score'''

def softmax(tensor):
  "Compute softmax values for each sets of scores in tensor."
  e = np.exp(tensor)
  return e / e.sum()

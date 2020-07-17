import transformers
from transformers import *
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os

def run_rating(input_ids, attention_mask):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  output_dir = "./my_BERT"
  PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

  # model_loaded = BertModel.from_pretrained(output_dir)
  tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed

  class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
      super(SentimentClassifier, self).__init__()
      self.config = BertConfig.from_pretrained(output_dir)
      
      self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
      self.drop = nn.Dropout(p=0.3)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
      _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      output = self.drop(pooled_output)
      return self.out(output)

  class_names = [0,1,2,3,4]
  model_loaded = SentimentClassifier(len(class_names))

  checkpoint = torch.load(output_dir + "/rating_prediction_model.pt", map_location=torch.device('cpu'))

  model_loaded.load_state_dict(checkpoint['model_state_dict'])

  model_loaded = model_loaded.to(device)

  output = model_loaded(input_ids, attention_mask)
  output_list = output.tolist()[0]

  #calculating prediction prob
  antilog_list, converted_scored = [], []

  for value in output_list:
    antilog_list.append(np.exp(value))

  summation = sum(antilog_list)


  for value in antilog_list:
    converted_scored.append(round((value*100)/summation,2))


  #get final prediction
  _, prediction = torch.max(output, dim = 1)
  prediction = int(prediction)

  rating_result = {0:1, 1:2, 2:3, 3:4, 4:5}

  for key,value in rating_result.items():
    if prediction == key:
        return converted_scored, value
  
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig
import torch
import numpy as np
import pandas as pd
from torch import nn

def run(input_ids, attention_mask):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  output_dir = "./my_BERT"
  # output_dir = "/home/bsharma/testing/my_BERT"

  tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed
  PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

  class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
      super(SentimentClassifier, self).__init__()
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

  class_names = ['Negative', 'Positive']
  model_loaded = SentimentClassifier(len(class_names))

  checkpoint = torch.load(output_dir + "/sentiment_analysis_model.pt")
  model_loaded.load_state_dict(checkpoint['model_state_dict'])
  model_loaded = model_loaded.to(device)
  output = model_loaded(input_ids, attention_mask)
  _, prediction = torch.max(output, dim = 1)

  if prediction == 0:
    return 'Negative'
  return 'Positive'
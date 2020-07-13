import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# output_dir = "./my_BERT"
# model_loaded = BertModel.from_pretrained(output_dir)
# tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed

# class SentimentClassifier(nn.Module):
#   def __init__(self, n_classes):
#     super(SentimentClassifier, self).__init__()
#     # self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,output_hidden_states=True,output_attentions=True)
#     # self.config = BertConfig.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)
#     self.config = BertConfig.from_pretrained(output_dir)
#     # self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
#     self.bert = BertModel.from_pretrained(output_dir)
#     self.drop = nn.Dropout(p=0.3)
#     self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
#   def forward(self, input_ids, attention_mask):
#     _, pooled_output = self.bert(
#       input_ids=input_ids,
#       attention_mask=attention_mask
#     )

#     output = self.drop(pooled_output)
#     return self.out(output)

# # class_names = ['Negative', 'Positive']
# class_names = ['Negative', 'Positive']
# model_loaded = SentimentClassifier(len(class_names))
# model_loaded = model_loaded.to(device)


tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

pipe = pipeline('sentiment-analysis')


def run(review_text):

    results = pipe([review_text])

    for result in results:
        print(f"label:{result['label']}, with score:{round(result['score'], 4)}")
    

    # review_text = [review_text]
    # MAX_LEN= 160
    # for text in review_text:
    #   encoded_review = tokenizer.encode_plus(
    #     text,
    #     max_length=MAX_LEN,
    #     add_special_tokens=True,
    #     return_token_type_ids=True,
    #     pad_to_max_length=True,
    #     return_attention_mask=True,
    #     return_tensors='pt',
    #   )

    #   input_ids = encoded_review['input_ids'].to(device)
    #   attention_mask = encoded_review['attention_mask'].to(device)
    #   output = model_loaded(input_ids, attention_mask)
    #   _, prediction = torch.max(output, dim=1)
    #   # print("prediction", prediction)

    #   if prediction == 0:
    #     sentiment = "Negative"
    #   else:
    #     sentiment = "Positive"
    
    return result['label']
#!/usr/bin/env python

# # Importing the Dependencies


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re


# # Prepare the Model



tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


# # Calculate the Sentiment using the pretrained model



tokens = tokenizer.encode('I am bad boy. But I have a good herat.', return_tensors='pt')




sentiment = model(tokens)
int(torch.argmax(sentiment.logits[0]))+1



def calculate_sentiment(text):
    tokens = tokenizer.encode(text, return_tensors='pt')
    sentiment = model(tokens)
    return int(torch.argmax(sentiment.logits[0]))+1




calculate_sentiment('Does she love me ? I have not been a good pather to her. ')


# # Data Scraping for testing



def scrape_review_or_comments(url, reg, tag):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile(reg)
    results = soup.find_all(tag, {'class': regex})
    reviews_or_comments = [result.text for result in results]
    return reviews_or_comments




reviews_or_comments = scrape_review_or_comments('https://www.yelp.com/biz/mejico-sydney-2', '.*comment.*', 'p')


# # Load Data into the Data Frame



import numpy as np
import pandas as pd



df = pd.DataFrame(np.array(reviews_or_comments), columns=['review'])




df['sentiment'] = df['review'].apply(lambda k: calculate_sentiment(k[:512]))



df







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:00:57 2020

@author: kartikeysinha
"""

#import libraries
import pandas as pd
import numpy as np


#converting csv to txt files
import csv

with open('/Users/kartikeysinha/Desktop/projects/disaster tweets/train.csv','r') as csvin, open('train.txt', 'w') as tsvout:
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter='\t')

    for row in csvin:
        tsvout.writerow(row)


with open('/Users/kartikeysinha/Desktop/projects/disaster tweets/test.csv','r') as csvin, open('test.txt', 'w') as tsvout:
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter='\t')

    for row in csvin:
        tsvout.writerow(row)


#import train dataset
dataset_train = pd.read_csv('train.txt', sep='\t', quoting = 3)
dataset_train['keyword'] = dataset_train['keyword'].fillna('no_keyword')
dataset_train['location'] = dataset_train['location'].fillna('no_location')
dataset_train['text'] = dataset_train['text'].fillna('no_text')
dataset_train = dataset_train.dropna(axis=0, how='any')
X_train = dataset_train.iloc[:, -2].values
y_train = dataset_train.iloc[:, -1].values


#clean the data for training set
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords           #list of non-necessary words
from nltk.stem.porter import PorterStemmer  #used to convert words to stem word

columns = len(X_train)
corpus_train = []

for i in range(0, columns):
    tweet = re.sub('[^a-zA-Z]', ' ', X_train[i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    stops = set(stopwords.words('english'))
    tweet = [ps.stem(word) for word in tweet if not word in stops]
    tweet = ' '.join(tweet)
    corpus_train.append(tweet)


#Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X_train = cv.fit_transform(corpus_train).toarray()


#train model through Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20, criterion="entropy")
classifier.fit(X_train, y_train)


#import test dataset
dataset_test = pd.read_csv('test.txt', sep='\t', quoting = 3)
dataset_test['keyword'] = dataset_test['keyword'].fillna('no_keyword')
dataset_test['location'] = dataset_test['location'].fillna('no_location')
dataset_test['text'] = dataset_test['text'].fillna('no_text')
X_test = dataset_test.iloc[:, -1].values
X_test_ids = dataset_test.iloc[:, 0].values


#clean the data for training set
columns = len(X_test)
corpus_test = []

for i in range(0, columns):
    tweet = re.sub('[^a-zA-Z]', ' ', X_test[i])
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    stops = set(stopwords.words('english'))
    tweet = [ps.stem(word) for word in tweet if not word in stops]
    tweet = ' '.join(tweet)
    corpus_test.append(tweet)

X_test = np.array(corpus_test)


#predict
y_pred = classifier.predict(cv.transform(X_test))


#convert to CSV
df = pd.DataFrame({"Id" : X_test_ids, "target" : y_pred})
df.to_csv("submission.csv", index=False)


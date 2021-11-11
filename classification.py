#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: load training data
import datetime
start_time = datetime.datetime.now()

import evaluation
import testsets

import re
import nltk
import collections
import cmath
import string
import numpy as np
from nltk import word_tokenize
from nltk import pos_tag
from nltk import WordNetLemmatizer
from nltk import ngrams
from collections import Counter
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.stem.porter import*
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

nltk.download('punkt')
nltk.download('stopwords')

stoplist = set(stopwords.words("english"))

class preprocessing:
    def __init__(self,filename):
        #In both train and test parts, whole_ID and whole_sentiment store the "ID" and "sentiment", whole_text stores the text after preprocessing.
        self.whole_ID = []
        self.whole_sentiment = []
        self.whole_text = []
        self.filename = filename
        self.wordDcharacteristic = {}
        self.postext = []
        self.negtext = []
        self.neutext = []

    def process(self):
        #set up a dictionary map ID to sentence.
        IDDsentence = {}
        #set up a dictionary map ID to sentiment.
        IDDsentiment = {}
        Dword_tag = {}
        with open(self.filename,"r") as f:
            for row in f:
                #Firstly, use regular expressions to change URLs into "URLLINK", since URLs need "http" to be the signal
                row = re.sub(r"http\S+", "urllink", row)
                #Next, replace user mentions with "USERMENTION", since user mentions need "@" to be the singal
                row = re.sub("(@[A-Za-z0-9_]+)", "usermention", row)
                #Replace :) to happy, replace :( to sad
                row = re.sub(r":\)","happy",row)
                row = re.sub(r":\(", "sad", row)
                #Make tokenize to separate ID sentiment and sentence
                tokens = word_tokenize(row)
                #Save sentiment for the dictionary
                ID = tokens[0]
                self.whole_ID.append(ID)
                sentiment = tokens[1]
                self.whole_sentiment.append(sentiment)
                IDDsentiment[ID] = sentiment
                line = []
                for n in range(2,len(tokens)):
                    #Delete the stopword can help to reduce time and increase accuracy.
                    if tokens[n] not in stoplist:
                        line.append(tokens[n])
                #Now, row is the whole sentence
                row = ' '.join(line)
                #Remove all non-alphanumeric characters except spaces
                row = re.sub(r'\\u|\\n|\\r',"",row)
                row = ' '.join(re.findall(r"[0-9A-Za-z]*", row))
                #Remove words with only 1 character
                row = re.sub(r'\b[a-zA-Z]\b',"",row)
                #Remove numbers that are fully made of digits
                row = re.sub(r'\b[0-9]+\b',"",row)
                #Replace uppercase word to "uppercase"+word
                flag = re.findall('[A-Z][A-Z]+',row)
                for i in range(0,len(flag)):
                    row = re.sub(flag[i], "uppercase"+flag[i],row)
                #Replace uppercase to lowercase
                row = row.lower()

                #Use WordNetLemmatizer to process, pos_tag to tag all token
                wordnet_lemmatizer = WordNetLemmatizer()
                tokens = word_tokenize(row)
                postag = []
                for word in tokens:
                    wordlist = [word]
                    if word in Dword_tag.keys():
                        postag.append((word,Dword_tag[word]))
                    else:
                        tagofword = pos_tag(wordlist)
                        tag = tagofword[0][1]
                        Dword_tag[word] = tag
                        postag.append((word,Dword_tag[word]))
                #if pos was not setted, then lemmatize may not be useful
                tokens = []
                for word, tag in postag:
                    if tag.startswith('NN'):
                        word = wordnet_lemmatizer.lemmatize(word, pos='n')
                        self.wordDcharacteristic[word] = 'n'
                    elif tag.startswith('VB'):
                        word = wordnet_lemmatizer.lemmatize(word, pos='v')
                        self.wordDcharacteristic[word] = 'v'
                    elif tag.startswith('JJ'):
                        word = wordnet_lemmatizer.lemmatize(word, pos='a')
                        self.wordDcharacteristic[word] = 'a'
                    elif tag.startswith('R'):
                        word = wordnet_lemmatizer.lemmatize(word, pos='r')
                        self.wordDcharacteristic[word] = 'r'
                    else:
                        word = word
                        self.wordDcharacteristic[word] = 'special'
                    tokens.append(word)
                sentence = ' '.join(tokens)
                self.whole_text.append(sentence)
                if sentiment == "positive":
                    self.postext += tokens
                if sentiment == "negative":
                    self.negtext += tokens
                if sentiment == "neutral":
                    self.neutext += tokens


#Preprocessing the train dataset, to get whole_ID, whole_sentiment and whole_text.
#train = preprocessing("small_train.txt")
train = preprocessing("twitter-training-data.txt")
train.process()
print("----------Start Feature----------")
# I use tfidf to be the feature selection.
# Firstly, I use CountVectorizer to be the vectorizer, but the result is not good.
# vectorizer = CountVectorizer()
# So I use TFidVectorizer to be the vectorizer. ngram_range set to (1,2). (2,3) is not useful on the training dataset.
# Since feature in the train dataset is so long, I set max_features = 5000.
vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=5000)
transformer = TfidfTransformer()
tfidf_score = transformer.fit_transform(vectorizer.fit_transform(train.whole_text))
tfidf = tfidf_score.A


# I try to use sentiwordnet to get pos_score and neg_score for every sentence
# But the time consuming is large, and the accuracy is around 0.2 which is too low.
# So I give up sentiwordnet, and try to write a lexicons dictionary base on the training set.
"""lexicons_score = []
for i in train.whole_text:
    tokens=nltk.word_tokenize(i)
    pscore = 0
    nscore = 0
    for i in tokens:
        if train.wordDcharacteristic[i] == 'n' and len(list(swn.senti_synsets(i,'n')))>0:
            pscore+=(list(swn.senti_synsets(i,'n'))[0]).pos_score() #positive score of a word
            nscore+=(list(swn.senti_synsets(i,'n'))[0]).neg_score() #negative score of a word
        if train.wordDcharacteristic[i] == 'v' and len(list(swn.senti_synsets(i,'v')))>0:
            pscore+=(list(swn.senti_synsets(i,'v'))[0]).pos_score()
            nscore+=(list(swn.senti_synsets(i,'v'))[0]).neg_score()
        if train.wordDcharacteristic[i] == 'a' and len(list(swn.senti_synsets(i,'a')))>0:
            pscore+=(list(swn.senti_synsets(i,'a'))[0]).pos_score()
            nscore+=(list(swn.senti_synsets(i,'a'))[0]).neg_score()
        if train.wordDcharacteristic[i] == 'r' and len(list(swn.senti_synsets(i,'r')))>0:
            pscore+=(list(swn.senti_synsets(i,'r'))[0]).pos_score()
            nscore+=(list(swn.senti_synsets(i,'r'))[0]).neg_score()
        elif len(list(swn.senti_synsets(i)))>0:
            pscore+=(list(swn.senti_synsets(i))[0]).pos_score()
            nscore+=(list(swn.senti_synsets(i))[0]).neg_score()
    lexicons_score.append([pscore,nscore])
lexicons = np.array(lexicons_score)"""




# Use word2vec as a tool of word embedding
# Base on the train set, I set up the feature inclued pos_score, neg_score and neu_score.
# pos_score, neg_score and neu_score are the frequency that the word appear in pos_text, neg_text and neu_text
# Using these three attributes as feature, the accuracy is around 0.4, but the working time is extremely long, over 3000s.

"""trainsetlexicons_score = []
wordDpos_score = {}
wordDneg_score = {}
wordDneu_score = {}
for i in train.whole_text:
    pos_score = 0
    neg_score = 0
    neu_score = 0
    tokens=nltk.word_tokenize(i)
    for j in tokens:
        if j in wordDpos_score.keys():
            pos_score += wordDpos_score[j]
            neg_score += wordDneg_score[j]
            neu_score += wordDneu_score[j]
        else:
            wordDpos_score[j] = train.postext.count(j)/(train.postext.count(j)+train.negtext.count(j)+train.neutext.count(j))
            wordDneg_score[j] = train.negtext.count(j)/(train.postext.count(j)+train.negtext.count(j)+train.neutext.count(j))
            wordDneu_score[j] = train.neutext.count(j)/(train.postext.count(j)+train.negtext.count(j)+train.neutext.count(j))
            pos_score += wordDpos_score[j]
            neg_score += wordDneg_score[j]
            neu_score += wordDneu_score[j]
    trainsetlexicons_score.append([pos_score,neg_score,neu_score])
trainsetlexicons = np.array(trainsetlexicons_score)
print(trainsetlexicons.shape)"""

# Then I change the count method into "nltk.FreqDist", which will give a dictionary contain the word and count.
# Time improve from 3000 to 123s.
trainsetlexicons_score = []
posD = nltk.FreqDist(train.postext)
negD = nltk.FreqDist(train.negtext)
neuD = nltk.FreqDist(train.neutext)
wordDpos_score = {}
wordDneg_score = {}
wordDneu_score = {}
for i in train.whole_text:
    pos_score = 0
    neg_score = 0
    neu_score = 0
    whole_score = 0
    tokens=nltk.word_tokenize(i)
    for j in tokens:
        if j in wordDpos_score.keys():
            pos_score += wordDpos_score[j]
            neg_score += wordDneg_score[j]
            neu_score += wordDneu_score[j]
        else:
            whole_score = int(posD[j])+int(negD[j])+int(neuD[j])
            wordDpos_score[j] = posD[j]/whole_score
            wordDneg_score[j] = negD[j]/whole_score
            wordDneu_score[j] = neuD[j]/whole_score
            pos_score += wordDpos_score[j]
            neg_score += wordDneg_score[j]
            neu_score += wordDneu_score[j]
    trainsetlexicons_score.append([pos_score,neg_score,neu_score])
trainsetlexicons = np.array(trainsetlexicons_score)



# New feature that combine the tfidf and lexicons that based on the trainset.
#tfidf_trainsetlexicons = np.hstack((tfidf,trainsetlexicons))
tfidf_trainsetlexicons = sparse.hstack((tfidf_score,trainsetlexicons))

for classifier in ['Bayes', 'Logistic', 'SVM']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'Bayes':
        print('Training ' + classifier)
        # TODO: extract features for training classifier1
        # TODO: train sentiment classifier1
        Bayes = MultinomialNB()
        # Use tfidf to be the feature selection.
        #Bayes.fit(tfidf_score,train.whole_sentiment)
        # Use sentiwordnet to be the feature selection
        #Bayes.fit(lexicons,train.whole_sentiment)
        # Use lexicons based on the trainset as feature.
        #Bayes.fit(trainsetlexicons,train.whole_sentiment)
        # Use tfidf_trainsetlexicons as feature.
        Bayes.fit(tfidf_trainsetlexicons,train.whole_sentiment)

    elif classifier == 'Logistic':
        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        # TODO: train sentiment classifier2
        Logistic = LogisticRegression()
        # Use tfidf to be the feature selection.
        Logistic.fit(tfidf_score,train.whole_sentiment)
        # Use sentiwordnet to be the feature selection
        #Logistic.fit(lexicons,train.whole_sentiment)
        # Use lexicons based on the trainset as feature.
        #Logistic.fit(trainsetlexicons,train.whole_sentiment)
        # Use tfidf_trainsetlexicons as feature.
        #Logistic.fit(tfidf_trainsetlexicons,train.whole_sentiment)

    elif classifier == 'SVM':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        # TODO: train sentiment classifier3
        # Use C=1 to avoid overfitting, kernel was set as linear. Since we have three classes, decision_function_shape was set as 'ovo'.
        SVM = SVC(C=1, kernel='linear',decision_function_shape='ovo')
        # Use tfidf to be the feature selection.
        SVM.fit(tfidf_score,train.whole_sentiment)
        # Use sentiwordnet to be the feature selection
        #SVM.fit(lexicons,train.whole_sentiment)
        # Use lexicons based on the trainset as feature.
        #SVM.fit(trainsetlexicons,train.whole_sentiment)
        # Use tfidf_trainsetlexicons as feature.
        #SVM.fit(tfidf_trainsetlexicons,train.whole_sentiment)

for testset in testsets.testsets:
    # TODO: classify tweets in test set
    # Preprocessing the testset
    test = preprocessing(testset)
    test.process()
    
    # Get tfidf score from testset
    new_tfidf_score = transformer.transform(vectorizer.transform(test.whole_text))
    new_tfidf = new_tfidf_score.A
    
    # Get new sentiwordnet lexicons
    """lexicons_score = []
    for i in test.whole_text:
        tokens=nltk.word_tokenize(i)
        pscore = 0
        nscore = 0
        for i in tokens:
            if test.wordDcharacteristic[i] == 'n' and len(list(swn.senti_synsets(i,'n')))>0:
                pscore+=(list(swn.senti_synsets(i,'n'))[0]).pos_score() #positive score of a word
                nscore+=(list(swn.senti_synsets(i,'n'))[0]).neg_score() #negative score of a word
            if test.wordDcharacteristic[i] == 'v' and len(list(swn.senti_synsets(i,'v')))>0:
                pscore+=(list(swn.senti_synsets(i,'v'))[0]).pos_score()
                nscore+=(list(swn.senti_synsets(i,'v'))[0]).neg_score()
            if test.wordDcharacteristic[i] == 'a' and len(list(swn.senti_synsets(i,'a')))>0:
                pscore+=(list(swn.senti_synsets(i,'a'))[0]).pos_score()
                nscore+=(list(swn.senti_synsets(i,'a'))[0]).neg_score()
            if test.wordDcharacteristic[i] == 'r' and len(list(swn.senti_synsets(i,'r')))>0:
                pscore+=(list(swn.senti_synsets(i,'r'))[0]).pos_score()
                nscore+=(list(swn.senti_synsets(i,'r'))[0]).neg_score()
            elif len(list(swn.senti_synsets(i)))>0:
                pscore+=(list(swn.senti_synsets(i))[0]).pos_score()
                nscore+=(list(swn.senti_synsets(i))[0]).neg_score()
        lexicons_score.append([pscore,nscore])
    new_lexicons = np.array(lexicons_score)"""

    # Get trainsetlexicons from test set.
    new_trainsetlexicons_score = []
    for i in test.whole_text:
        pos_score = 0
        neg_score = 0
        neu_score = 0
        tokens=nltk.word_tokenize(i)
        for j in tokens:
            if j in wordDpos_score.keys():
                pos_score += wordDpos_score[j]
                neg_score += wordDneg_score[j]
                neu_score += wordDneu_score[j]
        new_trainsetlexicons_score.append([pos_score,neg_score,neu_score])
    new_trainsetlexicons = np.array(new_trainsetlexicons_score)

    #new_tfidf_trainsetlexicons = np.hstack((new_tfidf,new_trainsetlexicons))
    new_tfidf_trainsetlexicons = sparse.hstack((new_tfidf_score,new_trainsetlexicons))

    
    print("----------Bayes Result----------")
    #In Bayes Result, I use the conbine of tfidf and lexicons based on train set. Because it is the best feature for Bayes.

    #sentiment = Bayes.predict(new_tfidf_score)
    #sentiment = Bayes.predict(new_lexicons)
    #sentiment = Bayes.predict(new_trainsetlexicons)
    sentiment = Bayes.predict(new_tfidf_trainsetlexicons)
    predictions = dict(zip(test.whole_ID,sentiment))
    evaluation.evaluate(predictions, testset, 'Bayes')
    evaluation.confusion(predictions, testset, 'Bayes')

    print("----------Logistic Result----------")
    #In Logistic Result, I use tfidf as feature for the best accuracy.

    sentiment = Logistic.predict(new_tfidf_score)
    #sentiment = Logistic.predict(new_lexicons)
    #sentiment = Logistic.predict(new_trainsetlexicons)
    #sentiment = Logistic.predict(new_tfidf_trainsetlexicons)
    predictions = dict(zip(test.whole_ID,sentiment))
    evaluation.evaluate(predictions, testset, 'Logistic')
    evaluation.confusion(predictions, testset, 'Logistic')

    print("----------SVM Result----------")
    #In SVM Result, I use tfidf as feature for the best accuracy.

    sentiment = SVM.predict(new_tfidf_score)
    #sentiment = SVM.predict(new_lexicons)
    #sentiment = SVM.predict(new_trainsetlexicons)
    #sentiment = SVM.predict(new_tfidf_trainsetlexicons)
    predictions = dict(zip(test.whole_ID,sentiment))
    evaluation.evaluate(predictions, testset, 'SVM')
    evaluation.confusion(predictions, testset, 'SVM')

end_time = datetime.datetime.now()
interval = (end_time-start_time).seconds
print(interval)

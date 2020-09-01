# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:10:52 2020

@author: Shaunak_Sensarma
"""


import io
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import nltk
import pandas as pd
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
import gensim

import numpy as np  
import random  
import string 
import re  

from sklearn.feature_extraction.text import TfidfVectorizer

html = urlopen('https://en.wikipedia.org/wiki/Web_Mining')
bs = BeautifulSoup(html, "html.parser")
c = bs.find('p').findAll(recursive=False)
s = bs.find('script').findAll(recursive=False)
for i in c:
    if not i in s:                  #to remove script tag contents
        a=i.get_text()

    #storing the contents into a file
    
    with io.open('index1.txt', "a", encoding="utf-8") as f:
        f.write(a)
f.close()


html = urlopen('https://en.wikipedia.org/wiki/Data_mining')
bs = BeautifulSoup(html, "html.parser")
c = bs.find('p').findAll(recursive=False)
s = bs.find('script').findAll(recursive=False)
for i in c:
    if not i in s:                  #to remove script tag contents
        a=i.get_text()

    #storing the contents into a file
    
    with io.open('index2.txt', "a", encoding="utf-8") as f:
        f.write(a)
f.close()


# Python program for insert and search 
# operation in a Trie 

class TrieNode: 
	
	# Trie node class 
	def __init__(self): 
		self.children = [None]*26

		# isEndOfWord is True if node represent the end of the word 
		self.isEndOfWord = False

class Trie: 
	
	# Trie data structure class 
	def __init__(self): 
		self.root = self.getNode() 

	def getNode(self): 
	
		# Returns new trie node (initialized to NULLs) 
		return TrieNode() 

	def _charToIndex(self,ch): 
		
		# private helper function 
		# Converts key current character into index 
		# use only 'a' through 'z' and lower case 
		
		return ord(ch)-ord('a') 


	def insert(self,key): 
		
		# If not present, inserts key into trie 
		# If the key is prefix of trie node, 
		# just marks leaf node 
		pCrawl = self.root 
		length = len(key) 
		for level in range(length): 
			index = self._charToIndex(key[level]) 

			# if current character is not present 
			if not pCrawl.children[index]: 
				pCrawl.children[index] = self.getNode() 
			pCrawl = pCrawl.children[index] 

		# mark last node as leaf 
		pCrawl.isEndOfWord = True

	

# driver function 
def main(): 
    
    keys = [] 
    
	# Input keys (use only 'a' through 'z' and lower case) 
    stop_words = set(stopwords.words('english')) 
    file1 = open("index1.txt") 
    line = file1.read()# Use this to read file content as a stream: 
    words1 = line.split() 
    file2= open("index2.txt")
    line2=file2.read()
    words2=line2.split()
    for r in words1 or r in words2: 
        if not r in stop_words: 
            r.lower()
            keys.append(r)
    
	# Trie object 
    t = Trie()
	# Construct trie
    for key in keys:
        t.insert(key)    
	 

if __name__ == '__main__': 
	main() 



#for predictive typing part.


from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
path = '1661-0.txt'
text = open(path).read().lower()
print('corpus length:', len(text))
tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)
unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))
WORD_LENGTH = 5
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])
print(prev_words[0])
print(next_words[0])
X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=2, shuffle=True).history

model.save('keras_next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))
model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))

def prepare_input(text):
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    for t, word in enumerate(text.split()):
        print(word)
        x[0, t, unique_word_index[word]] = 1
    return x
prepare_input("It is not a lack".lower())


def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def predict_completions(text, n=3):
    if text == "":
        return("0")
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [unique_words[idx] for idx in next_indices]


print("Enter the sentence...")
q=input()
print("correct sentence: ",q)
seq = " ".join(tokenizer.tokenize(q.lower())[0:5])
print("Sequence: ",seq)
print("next possible words: ", predict_completions(seq, 5))



#auto-correct-portion


import os.path
import collections
from operator import itemgetter

class Autocorrect(object):
    """
    Very simplistic implementation of autocorrect using ngrams.
    """
    def __init__(self, ngram_size=3, len_variance=1):
        self.ngram_size = ngram_size
        self.len_variance = len_variance

        wordfile = os.path.join(os.path.dirname(__file__), "triedoc.txt")
        self.words = set(open(wordfile).read().splitlines())

        # create dictionary of ngrams and the words that contain them
        self.ngram_words = collections.defaultdict(set)
        for word in self.words:
            for ngram in self.ngrams(word):
                self.ngram_words[ngram].add(word)
        print ("Generated %d ngrams from %d words" % (len(self.ngram_words), len(self.words)))

    def lookup(self, word):
        "Return True if the word exists in the dictionary."
        return word in self.words

    def ngrams(self, word):
        "Given a word, return the set of unique ngrams in that word."
        all_ngrams = set()
        for i in range(0, len(word) - self.ngram_size + 1):
            all_ngrams.add(word[i:i + self.ngram_size])
        return all_ngrams

    def suggested_words(self, target_word, results=5):
        "Given a word, return a list of possible corrections."
        word_ranking = collections.defaultdict(int)
        possible_words = set()
        for ngram in self.ngrams(target_word):
            words = self.ngram_words[ngram]
            for word in words:
                # only use words that are within +-LEN_VARIANCE characters in 
                # length of the target word
                if len(word) >= len(target_word) - self.len_variance and \
                   len(word) <= len(target_word) + self.len_variance:
                    word_ranking[word] += 1
        # sort by descending frequency
        ranked_word_pairs = sorted(word_ranking.items(), key=itemgetter(1), reverse=True)
        return [word_pair[0] for word_pair in ranked_word_pairs[0:results]]


if __name__ == '__main__':
    autocorrect = Autocorrect()
    while True:
        print("Enter a word: ")
        word=input().lower()
        
        if autocorrect.lookup(word):
            print ("Looks good to me!")
        else:
            suggestions = autocorrect.suggested_words(word)
            print ("Maybe you meant: %s" % ", ".join(suggestions))


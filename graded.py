# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:36:20 2020

@author: Shaunak_Sensarma
"""

import io 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
stop_words = set(stopwords.words('english')) 
stop_words.append('.',',','a','they','the','his','so','and','were','from','that','of','in','only','with','to')
file1 = open("ML.txt") 
line = file1.read()# Use this to read file content as a stream: 
words = line.split() 
for r in words: 
    if not r in stop_words: 
        appendFile = open('filteredML.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close() 
num_words = 0
 
with open('filteredML.txt', 'r') as f:
    for line in f:
        words = line.split()
        num_words += len(words)
print("Number of words in First ML File:")
print(num_words)


file1 = open("ML.txt") 
line = file1.read()# Use this to read file content as a stream: 
words = line.split() 
for r in words: 
    if r in stop_words: 
        appendFile = open('filML.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close()
from collections import Counter
def word_count(fname):
        with open(fname) as f:
                return Counter(f.read().split())
print("Frequency of stop-words in the ML file :",word_count("filML.txt"))

file2 = open("AI.txt") 
line = file2.read()# Use this to read file content as a stream: 
words = line.split() 
for r in words: 
    if not r in stop_words: 
        appendFile = open('filteredAI.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close() 
num_words = 0
 
with open('filteredAI.txt', 'r') as f:
    for line in f:
        words = line.split()
        num_words += len(words)
print("Number of words in Second AI File:")
print(num_words)

file2 = open("AI.txt") 
line = file2.read()# Use this to read file content as a stream: 
words = line.split() 
for r in words: 
    if r in stop_words: 
        appendFile = open('filAI.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close()
from collections import Counter
def word_count(fname):
        with open(fname) as f:
                return Counter(f.read().split())
print("Frequency of stop-words in the AI file :",word_count("filAI.txt"))

file1 = open("filteredML.txt") 
line = file1.read()# Use this to read file content as a stream: 
words1 = line.split() 
file2 = open("filteredAI.txt") 
line = file2.read()# Use this to read file content as a stream: 
words2 = line.split() 
for r in words1: 
    if r in words2: 
        appendFile = open('finaltext.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close()
print()
print("Common terms in both text are.")
fopen3 = open("finaltext.txt", 'r')
for line in fopen3:
    print(line)

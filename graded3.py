# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:44:34 2020

@author: Shaunak_Sensarma
"""
import io 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 

stop_words = set(stopwords.words('english')) 
file1 = open("doc1.txt") 
line = file1.read()# Use this to read file content as a stream: 
words1 = line.split() 
file2 = open("doc2.txt") 
line = file2.read()# Use this to read file content as a stream: 
words2 = line.split() 
for r in words1: 
    if r in words2: 
        appendFile = open('newfile.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close()
print()


file1 = open("newfile.txt") 
line = file1.read()# Use this to read file content as a stream: 
words = line.split() 
for r in words: 
    if not r in stop_words: 
        appendFile = open('index.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close() 

print("Common terms in both text after removing stop words are.")
print()
fopen3 = open("index.txt", 'r')
for line in fopen3:
    print(line)
    
num_words=0
with open('index.txt', 'r') as f:
    for line in f:
        words = line.split()
        num_words += len(words)
print("Number of words in Index File:")
print(num_words)
print()
print("Applying Stemming...")
print()
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language='english')
new_text = "Python Python Python standard available Python Python Python Python Python Python standard Python Standard Python Python Python Python Python Python Standard"
words = word_tokenize(new_text)

for token in words:
    print(token + ' --> ' + stemmer.stem(token))
    appendFile = open('indexS.txt','a') 
    appendFile.write(" "+stemmer.stem(token)) 
    appendFile.close() 


print()
print("Applying Lemmitization")
print()
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer() 
new_text = "Python Python Python standard available Python Python Python Python Python Python standard Python Standard Python Python Python Python Python Python Standard"
words = word_tokenize(new_text)

tagged = nltk.pos_tag(words) 


for w in words:
    print(w, " - ", ps.stem(w), " - ", lemmatizer.lemmatize(w))
    appendFile = open('indexL.txt','a') 
    appendFile.write(" "+lemmatizer.lemmatize(w)) 
    appendFile.close() 

numStem=0
with open('indexS.txt', 'r') as f:
    for line in f:
        words = line.split()
        numStem += len(words)
print()
print("Count after stemming:")
print(numStem)

numLem=0
with open('indexL.txt', 'r') as f:
    for line in f:
        words = line.split()
        numLem += len(words)
print()        
print("Count after Lemmitization:")
print(numLem)

if(numStem>=numLem):
    file1 = open("indexS.txt") 
    line = file1.read()# Use this to read file content as a stream: 
    words = line.split() 
    for r in words:  
        appendFile = open('final-index.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close() 
else:
    file1 = open("indexF.txt") 
    line = file1.read()# Use this to read file content as a stream: 
    words = line.split() 
    for r in words:  
        appendFile = open('final-index.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close() 
print()        
print("Displaying Contents of final-index document in POS form")
fopen3 = open("final-index.txt", 'r')
for line in fopen3:
    print(line)
    
    
    
    
    
    
    
    
    
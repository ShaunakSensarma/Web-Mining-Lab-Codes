import requests
from bs4 import BeautifulSoup
from collections import deque
print("No of depth= 5")
print("no of threas= 5")
print()
visited = set(["https://shaunaksensarma.github.io"])
dq = deque([["https://shaunaksensarma.github.io", "", 0]])
max_depth = 5
max_workers=5
while dq:
    base, path, depth = dq.popleft()
    #                         ^^^^ removing "left" makes this a DFS (stack)

    if depth < max_depth:
        try:
            soup = BeautifulSoup(requests.get(base + path).text, "html.parser")

            for link in soup.find_all("a"):
                href = link.get("href")

                if href not in visited:
                    visited.add(href)
                    print("  " * depth + f"at depth {depth+1}: {href}")

                    if href.startswith("http"):
                        dq.append([href, "", depth+1 ])
                    else:
                        dq.append([base, href, depth + 1])
        except:
            pass
def crawlingl():
	homDir = '.'
	if not os.path.exists(homDir):
		os.mkdir(homDir)
	stack = [url]
	with concurrent.futures.ThreadPoolExecutor(max_workers = 5) as executor:
		while len(stack) != 0:# DFS
			topUrl = stack.pop()
			urltoken = topUrl.split('/')
			fdir = homDir + '/' + topUrl[topUrl.find('Programming/'):]#dir in file system
			if urltoken[-1] == '':#if path
				print(topUrl)
				if not os.path.exists(fdir):
					os.mkdir(fdir)
				response = urlreq.urlopen(topUrl)
				soup = BeautifulSoup(response.read(), 'lxml')
				for link in soup.find_all('a'):
					href = link.get('href')
                
					if not href.startswith('/') and not href.startswith('?'):
						stack.append(topUrl + link.get('href'))
			else:
				executor.submit(download, fdir, url)


#2nd part of the question to implement crawler properties


#extracting content
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


html = urlopen('https://shaunaksensarma.github.io/')
bs = BeautifulSoup(html, "html.parser")
c = bs.find('p').findAll(recursive=False)
s = bs.find('script').findAll(recursive=False)
for i in c:
    if not i in s:                  #to remove script tag contents
        a=i.get_text()

    #storing the contents into a file
    
    with io.open('code.txt', "a", encoding="utf-8") as f:
        f.write(a)
    f.close()

#counting total terms and term frequency
    
stop_words = set(stopwords.words('english')) 

file1 = open("code.txt") 
line = file1.read()# Use this to read file content as a stream: 
words = line.split() 
for r in words: 
    if not r in stop_words: 
        appendFile = open('filter.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close() 
num_words = 0
 
with open('filter.txt', 'r') as f:
    for line in f:
        words = line.split()
        num_words += len(words)

print()
print("Number of terms after stopwords removal..:")
print(num_words)
print()
print()

from collections import Counter
def word_count(fname):
        with open(fname) as f:
                return Counter(f.read().split())
print("Frequency of terms :")
print()
print(word_count("code.txt"))
print()

#to add special characters as stopwords.

newStopWords = ['@','.','&','%','!','$','^','-','_','/',',']
#stopwords.extend(newStopWords)
file1 = open("code.txt") 
line = file1.read()# Use this to read file content as a stream: 
words = line.split() 
for r in words: 
    if not r in stop_words: 
        appendFile = open('codeNew.txt','a') 
        appendFile.write(" "+r) 
        appendFile.close() 

    
print()
print()
print("After Stemmming and lemmatization...") 
print()
   
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer(language='english')
term=[]
tokens = ["Web","mining","application","data","mining","techniques","discover","patterns","World","Wide","Web","name","proposes","information","gathered","mining","web","makes","utilization","automated","apparatuses","reveal","extricate","data","servers","web2","reports","permits","organizations","get","both","organized","unstructured","information","browser","activities","server","logs","website","link","structure","page","content","different","sources"]

df = pd.DataFrame()

stem = []
for token in tokens:
   stem.append(stemmer.stem(token))
    
countterm=0 


import spacy
sp = spacy.load('en_core_web_sm')
sentence7 = sp(u'Web mining application data mining techniques discover patterns World Wide Web name proposes information gathered mining web makes utilization automated apparatuses reveal extricate data servers web2 reports permits organizations get both organized unstructured information browser activities server logs website link structure page content different sources')

lemma=[]

for word in sentence7:
    term.append(word)
    lemma.append(word.lemma_)
    countterm=countterm+1


df = pd.DataFrame()
df['word']=term
df['Stemmed']  = stem
df['Lemmatized']=lemma
print(df)
print()



sent=[]
stop=[]
stop1=[]
with open('code.txt', 'r') as f:
    for line in f:
        sent.append(line)
        words = line.split()
        for r in words: 
            if r in stop_words: 
                stop1.append(r)
        stop.append(stop1)

df1 = pd.DataFrame()
df1['Sentence']=sent
df1['Stop Words']  = stop
print(df1)

 

#inverted index file



def createDictionary():

    wordsAdded = {}
    cwd = os.getcwd()
    os.chdir('C:\\Users\\Shaunak_Sensarma\\Desktop\\Inverted-Index-Python-master\\text-files')
    fileList = os.listdir(os.getcwd())

    for file in fileList:

        with open(file, 'r') as f:

            words = f.read().lower().split()

            for word in words:

                if word[-1] in [',', '!', '?', '.']:
                    word = word[:-1]
                if word not in wordsAdded.keys():
                    wordsAdded[word] = [f.name]

                else:
                    if file not in wordsAdded[word]:
                        wordsAdded[word] += [f.name]

    return wordsAdded, cwd


def writeToFile(words, cwd):
    os.chdir(cwd)
    with open('index-file.txt', 'w') as indexFile:

        for word, files in words.items():
            indexFile.write(word + " ")
            for file in files:
                indexFile.write(file[:file.find(".txt")] + " ")

            indexFile.write(f'{len(files)}\n')


writeToFile(*createDictionary())

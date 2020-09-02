import requests
from bs4 import BeautifulSoup
urlw = "https://en.wikipedia.org/wiki/Web_mining"
urld = "https://en.wikipedia.org/wiki/Data_mining"
urlml = "https://en.wikipedia.org/wiki/Machine_learning"
urlai = "https://en.wikipedia.org/wiki/Artificial_intelligence"
urlm = "https://en.wikipedia.org/wiki/Mining"
query = "Mining large volume of data"
web_text = requests.get(urlw).text.lower()
data_text = requests.get(urld).text.lower()
ml_text = requests.get(urlml).text.lower()
ai_text = requests.get(urlai).text.lower()
m_text = requests.get(urlm).text.lower()

web_content = BeautifulSoup(web_text,'html.parser')
data_content = BeautifulSoup(data_text,'html.parser')
ml_content = BeautifulSoup(ml_text,'html.parser')
ai_content = BeautifulSoup(ai_text,'html.parser')
m_content = BeautifulSoup(m_text,'html.parser')

for s in web_content.select('script'):
    s.extract()
for t in data_content.select('script'):
    t.extract()
for t in ml_content.select('script'):
    t.extract()
for t in ai_content.select('script'):
    t.extract()
for t in m_content.select('script'):
    t.extract()

web_content=web_content.text.lower()
data_content=data_content.text.lower()
ml_content=ml_content.text.lower()
ai_content=ai_content.text.lower()
m_content=m_content.text.lower()

print()
print("Query: Mining large volume of data")
print()
import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
special_chars = ["·",",","-","*",".","/","`","~","–","'\'", "!","@","#","_", "£","‘",
                 "$","'","``","(",")","''","=","[","]","{","}","?","<",">",";",":","’",
                 "”","“", "%", "&", "+", "-" ,"^" ,"$", "﹩", "＄", "€", "£", "₤", "₾", 
                 "Ⴊ", "ⴊ", "ლ", "₵", "¥", "￥" ,"﷼", "฿", "|", ":","•"]
stop_words=list(stop_words)
stop_words.extend(special_chars)
web_word_tokens = word_tokenize(web_content)
data_word_tokens = word_tokenize(data_content)
ml_word_tokens = word_tokenize(ml_content)
ai_word_tokens = word_tokenize(ai_content)
m_word_tokens = word_tokenize(m_content)
q_tokens = word_tokenize(query)

filtered_web_content = [w for w in web_word_tokens if w not in stop_words]
filtered_data_content = [w for w in data_word_tokens if w not in stop_words]
filtered_ml_content = [w for w in ml_word_tokens if w not in stop_words]
filtered_ai_content = [w for w in ai_word_tokens if w not in stop_words]
filtered_m_content = [w for w in m_word_tokens if w not in stop_words]
filtered_query = [w for w in q_tokens if w not in stop_words]
print()
print()

full_doc = set(filtered_web_content).union(set(filtered_data_content)).union(set(filtered_ml_content))
full_doc.union(set(filtered_ai_content)).union(set(filtered_m_content)).union(set(filtered_query))
full_doc = list(full_doc)
len(full_doc)
print()

import nltk
nltk.download('wordnet')

from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
stemmed_data = []
lemmatized_data = []
for i in full_doc:
    stemmed_data.append(stemmer.stem(i))
    lemmatized_data.append(lemmatizer.lemmatize(i))
    
len(lemmatized_data)

import pandas as pd
bow_web=[]
bow_data=[]
bow_ml=[]
bow_ai=[]
bow_m=[]
bow_q =[]
for i in full_doc:
    bow_web.append(filtered_web_content.count(i))
    bow_data.append(filtered_data_content.count(i))
    bow_ml.append(filtered_ml_content.count(i))
    bow_ai.append(filtered_ai_content.count(i))
    bow_m.append(filtered_m_content.count(i))
    bow_q.append(filtered_query.count(i))

bow_dict = {
    "Terms": full_doc,
    "Web mining doc": bow_web,
    "Data mining doc":bow_data,
    "ML doc":bow_ml,
    "AI doc":bow_ai,
    "Mining doc":bow_m,
    "query":bow_q
}
bow_df = pd.DataFrame(bow_dict, columns=[i for i in bow_dict])
#bow_df[(bow_df["Web mining doc"] >= 2) & (bow_df["Data mining doc"] <= 10)]
print()
print()
print(bow_df)
print()
print()


import string, numpy
stemmer = nltk.stem.porter.PorterStemmer()
def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def StemNormalize(text):
    return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import CountVectorizer
def tostr(lst): 
    return ' '.join(lst)

docs = [tostr(filtered_web_content),tostr(filtered_data_content),tostr(filtered_ml_content),tostr(filtered_ai_content),tostr(filtered_m_content),query]
LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(docs)
print(LemVectorizer.vocabulary_)
print()
print()
 
tf_matrix = LemVectorizer.transform(docs).toarray()
print(tf_matrix)
print()
print()
from sklearn.feature_extraction.text import TfidfTransformer
tfidfTran = TfidfTransformer(norm='l2')
tfidfTran.fit(tf_matrix)
print(tfidfTran.idf_)
print()
print()
tfidf_matrix = tfidfTran.transform(tf_matrix)
tfidf = tfidf_matrix.toarray()
print(tfidf)
print()
print()
l = len(tfidf)
for i in range(l):
    t=0
    for j in range(len(tfidf[i])):
        t += tfidf[i][i] ** 2
    t = t ** 0.5
    for k in range(len(tfidf[i])):
        tfidf[i][k] /= t
        
cosine = [0]*6

for i in range (len (tfidf)):
    for j in range (len(tfidf[i])):
        cosine[i] += tfidf[i][j] * tfidf[5][j]
    cosine[i] = round(cosine[i], 5)

for i in range (len(tfidf)):
    print("Cosine similarity of Doc{}: ".format(i) + str (cosine[i]))
print()
print()    
eu = [0]*6
for i in range(len(tfidf)):
    for j in range(len(tfidf[i])):
        eu[i] += (tfidf[i][j] - tfidf[5][j])**2
    eu[i] = round(eu[i]**0.5 , 5)
for i in range (len(tfidf)):
    print("Euclidean distance of Doc{}: ".format(i) + str(eu[i]))
print()
print()

cosine_docs = cosine[:-1]
cosine_docs.sort()
d = ["Doc 0","Doc 1","Doc 2","Doc 3","Doc 4"]
rank = list(zip(d,cosine_docs))
rank = sorted(rank, key = lambda x: x[1],reverse = True)
#sorted based on rank
for i in range(len(rank)):
    print("{}. {}".format(i+1, rank[i]))
print()
print()
cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print(cos_similarity_matrix)
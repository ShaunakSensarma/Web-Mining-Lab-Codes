import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 

stop_words = set(stopwords.words('english'))

from nltk.corpus import wordnet 

example_sent = input("Enter text : ")
print()
print("Given text is....")
print(example_sent)
print()

words = word_tokenize(example_sent)
wrd=words

words = [w for w in words if not w in stop_words] 

synonyms = []
count=[]

for w in words:
    for syn in wordnet.synsets(w):
        for lm in syn.lemmas():
            synonyms.append(lm.name())#adding into synonyms
    count.append(synonyms) 
    synonyms = []
b=0
for i in range(0,3):
    w=""
    a=0
    for k in wrd:
        
        if k in stop_words:
            w=w+k+" "
        else:
            if(len(count[a])<2) or (len(count[a][b])<2):
                w=w+k+" "
            else:    
                w=w+count[a][b]+" "
            a=a+1
    b=b+1
    print(w)        
        
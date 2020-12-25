import nltk                                 #Main library for preprocessing
import urllib.request                       #Library for load and read url
import spacy                                #Library for solving NER, and Noun Phrase Detection
import pandas as pd                         #Library for storing the result of ranking

from bs4 import BeautifulSoup               #Library for parsing html
from collections import Counter             #Library for counting the term and saving to the dictionary
from nltk import re, word_tokenize          #Library for regular expression and tokenization    
from nltk.corpus import stopwords           #Library for removing stopwords
from nltk.tokenize import RegexpTokenizer   #Library for tokenization
from sklearn.feature_extraction.text import TfidfVectorizer     #ibrary for tfidf

#Setting spacy into english language
nlp = spacy.load('en_core_web_sm')
      
#Setting stopword in english
stop_words = set(stopwords.words("english"))

#Reading URL and parsing it with html parser
file1 = urllib.request.urlopen("http://irsg.bcs.org/ksjaward.php").read()
file2 = urllib.request.urlopen("http://csee.essex.ac.uk/staff/udo/index.html").read()
soup1 = BeautifulSoup(file1, 'html.parser')
soup2 = BeautifulSoup(file2, 'html.parser')

#Retaining all the content from metatags
for metatag in soup1.find_all('meta'):
    metatags1 = metatag.get('content')
for metatag in soup2.find_all('meta'):
    metatags2 = metatag.get('content')

#Stripping out all javascript code
for tag in soup1.find_all('script'):
    tag.replaceWith('')
for tag in soup2.find_all('script'):
    tag.replaceWith('')

#Eliminating many new lines int the text   
removemanyline1 = re.sub(r'\n\s*\n', r',\n', soup1.get_text().strip(), flags=re.M)
removemanyline2 = re.sub(r'\n\s*\n', r',\n', soup2.get_text().strip(), flags=re.M)
            
#Substituting new line into ,
substitute1 = re.sub(r'\n', r',', removemanyline1.strip(), flags=re.M)
substitute2 = re.sub(r'\n', r',', removemanyline2.strip(), flags=re.M)
            
#removing more than 2 whitespaces
removewhitespace1 = re.sub(r'\s{2,}', r'', substitute1.strip(), flags=re.M)
removewhitespace2 = re.sub(r'\s{2,}', r'', substitute2.strip(), flags=re.M)
            
#cleaning data from (, ), "
content1 = removewhitespace1.strip().replace("(","").replace(")","").replace('"','"')
content2 = removewhitespace2.strip().replace("(","").replace(")","").replace('"','"')
            
#tokenization and removing punctuation on metatags by stackoverflow
m_tokenizer1 = RegexpTokenizer(r'\w+')
metatag_tokens1 = m_tokenizer1.tokenize(metatags1)
m_tokenizer2 = RegexpTokenizer(r'\w+')
metatag_tokens2 = m_tokenizer2.tokenize(metatags2)
         
#tokenization and detecting only alphanumeric characthers on content
c_tokenizer1 = RegexpTokenizer(r'\w+')
content_tokens1 = c_tokenizer1.tokenize(content1)
c_tokenizer2 = RegexpTokenizer(r'\w+')
content_tokens2 = c_tokenizer2.tokenize(content2)

#removing stopwords on metatag
filtered_metatag1 = [word.lower() for word in metatag_tokens1 if not word in stop_words]
filtered_metatag2 = [word.lower() for word in metatag_tokens2 if not word in stop_words]
            
#removing stopwords on data
filtered_content1 = [word.lower() for word in content_tokens1 if not word in stop_words]
filtered_content2 = [word.lower() for word in content_tokens2 if not word in stop_words]
           
#Initialising spacy data
doc = nlp(content1)
doc = nlp(content2)
            
#joining element for noun phrase
chunk1 = (list(doc.noun_chunks))
chunk_string1 = ','.join(str(v) for v in chunk1)
noun1 = chunk_string1.split(',')
chunk2 = (list(doc.noun_chunks))
chunk_string2 = ','.join(str(v) for v in chunk2)
noun2 = chunk_string2.split(',')

#Named entity recognition 
ner1 = (list(doc.ents))
ner_string1 = ','.join(str(v) for v in ner1)
entity1 = ner_string1.split(',')
ner2 = (list(doc.ents))
ner_string2 = ','.join(str(v) for v in ner2)
entity2 = ner_string2.split(',')

#Combining all the filtered contents: metatag, content, entity, noun phrase
document1 = filtered_metatag1 + filtered_content1 + entity1 + noun1
document2 = filtered_metatag2 + filtered_content2 + entity2 + noun2

#TF-IDF
#Counting occurrences each term in a document
nterm1 = Counter()
for word in document1:
    nterm1[word] += 1

nterm2 = Counter()
for word in document2:
    nterm2[word] += 1

#obtaining list of vocabulary documents
dic1 = set(nterm1)
dic2 = set(nterm2)

#merging all vocabularies from both documents
wordset = set(nterm1).union(set(nterm2))

#Setting the all vocabularies into matrix and assigning 0 value for each content
worddic1 = dict.fromkeys(wordset, 0)
worddic2 = dict.fromkeys(wordset, 0)

#Counting how many occurrences for each term in each document
for word in dic1:
    worddic1[word] += 1
for word in dic2:
    worddic2[word] += 1

#Building a function for counting Term Frequency
def tf(worddic, dic):
    tfdic = {}
    #Counting how many terms in a document 
    diccount = len(dic)
    for word, count in worddic.items():
        #formula is an number of a term's occurrences divided by total number for all terms in a document
        tfdic[word] = count / float(diccount)
    return tfdic

#Assigning the function to a variable a variable for each document
tf1 = tf(worddic1, dic1)
tf2 = tf(worddic2, dic2)

#Building a function for counting IDF
def idf(doclist):
    import math
    idfdic = {}
    #Counting how many terms in both documets
    n = len(doclist)
    idfdic = dict.fromkeys(doclist[0].keys(), 0)
    for doc in doclist:
        for word, val in doc.items():
            if val > 0:
                idfdic[word] += 1
    for word, val in idfdic.items():
        #Calculating total number of documents / number of documents with that particular term
        idfdic[word] = math.log(n / float(val))
    return idfdic

#Assigning the function to a variable a variable for each document
idfs = idf([worddic1, worddic2])

#Building a function for calculating TFIDF
def calculate_tfidf(tf, idfs):
    tfidf = {}
    for word, val in tf.items():
        #Calculating tfidf
        tfidf[word] = val * idfs[word]
    return tfidf

#Assigining the function into a variable for each document
tfidf1 = calculate_tfidf(tf1, idfs)
tfidf2 = calculate_tfidf(tf2, idfs)

#Storing data in pandas
df = pd.DataFrame([tfidf1, tfidf2])
#Transposing the result
transpose = df.T
#wordset
print(transpose)



#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes

# ##### Bayes’s Theorem
# 
# According to the Wikipedia, In probability theory and statistics,** Bayes’s theorem** (alternatively *Bayes’s law* or *Bayes’s rule*) describes the probability of an event, based on prior knowledge of conditions that might be related to the event.
# Mathematically, it can be written as:
# 
# ![formula.jpeg](attachment:formula.jpeg)
# 
# Where A and B are events and P(B)≠0
# * P(A|B) is a conditional probability: the likelihood of event A occurring given that B is true.
# * P(B|A) is also a conditional probability: the likelihood of event B occurring given that A is true.
# * P(A) and P(B) are the probabilities of observing A and B respectively; they are known as the marginal probability.
# 

# Let’s understand it with the help of an example:
# 
# **The problem statement:**
# 
# You are planning a picnic today, but the morning is cloudy
# 
# Oh no! 50% of all rainy days start off cloudy!
# But cloudy mornings are common (about 40% of days start cloudy)
# And this is usually a dry month (only 3 of 30 days tend to be rainy, or 10%)
# What is the chance of rain during the day?
# 
# We will use Rain to mean rain during the day, and Cloud to mean cloudy morning.
# 
# The chance of Rain given Cloud is written P(Rain|Cloud)
# 
# So let's put that in the formula:
# 
# $P(Rain|Cloud) = \frac{P(Rain)*P(Cloud|Rain)} {P(Cloud)}$          
#                       
#  
# 
# - P(Rain) is Probability of Rain = 10%
# - P(Cloud|Rain) is Probability of Cloud, given that Rain happens = 50%
# - P(Cloud) is Probability of Cloud = 40%
# 
# $P(Rain|Cloud) =  \frac{(0.1 x 0.5)} {0.4}   = .125$
# 
# Or a 12.5% chance of rain. Not too bad, let's have a picnic!

# **Naïve:** It is called Naïve because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. Such as if the fruit is identified on the bases of color, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identify that it is an apple without depending on each other.<br>
# **Bayes:** It is called Bayes because it depends on the principle of Bayes' Theorem

# # Problem statement

# Spam filtering using naive Bayes classifiers in order to predict whether a new mail based on its content, can be categorized as spam or not-spam.

# ### Data processing using panda library

# In[1]:


# Import the required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string
import matplotlib.pyplot as plt


# In[2]:


# Load the dataset

data = pd.read_csv("spam.tsv",,sep='\t'names=['Class','Message'])
data.head(8) # View the first 8 records of our dataset


# In[3]:


# to view the first record
data.loc[:0]


# In[4]:


# Summary of the dataset
data.info()


# In[5]:


data.describe()


# In[6]:


# create a column to keep the count of the characters present in each record
data['Length'] = data['Message'].apply(len)


# In[7]:


data['Length']


# In[8]:


# view the dataset with the column 'Length' which contains the number of characters present in each mail
data.head(10)


# In[9]:


## The mails are categorised into 2 classes ie., spam and ham. 
# Let's see the count of each class
data.groupby('Class').count()


# ### Data Visualization

# In[10]:


data['Length'].describe() # to find the max length of the message. 


# See what we found, A 910 character long message. Let's use masking to find this message:

# In[11]:


data['Length']==910


# In[12]:


# the message that has the max characters
data[data['Length']==910]['Message']


# In[13]:


# view the message that has 910 characters in it
data[data['Length']==910]['Message'].iloc[0]


# In[14]:


data[data['Length']==2]['Message']


# In[15]:


# View the message that has min characters
data[data['Length']==2]['Message'].iloc[0]


# In[ ]:





# ### Text Pre-Processing

# In[16]:


# creating an object for the target values
dObject = data['Class'].values
dObject


# In[17]:


# Lets assign ham as 1
data.loc[data['Class']=="ham","Class"] = 1


# In[18]:


# Lets assign spam as 0
data.loc[data['Class']=="spam","Class"] = 0


# In[19]:


dObject2=data['Class'].values
dObject2


# In[20]:


data.head(8)


# First removing punctuation. We can just take advantage of Python's built-in string library to get a quick list of all the possible punctuation:

# In[21]:


# the default list of punctuations
import string

string.punctuation


# In[22]:


# Why is it important to remove punctuation?
'Best in the World' == 'Best in the World'
"This message is spam" == "This message is spam."


# In[23]:


# Let's remove the punctuation

def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    return text

data['text_clean'] = data['Message'].apply(lambda x: remove_punct(x))

data.head()


# __Tokenization__ (process of converting the normal text strings in to a list of tokens(also known as lemmas)).

# In[24]:


# original text and cleaned text
data.head(8)


# Now we need to convert each of those messages into a vector the SciKit Learn's algorithm models can work with and machine learning model which we will gonig to use can understand.

# In[25]:


# Countvectorizer is a method to convert text to numerical data. 

# Initialize the object for countvectorizer 
CV = CountVectorizer(stop_words="english")


# [Stopwords are the words in any language which does not add much meaning to a sentence. They are the words which are very common in text documents such as a, an, the, you, your, etc. The Stop Words highly appear in text documents. However, they are not being helpful for text analysis in many of the cases, So it is better to remove from the text. We can focus on the important words if stop words have removed.]

# In[26]:


# Splitting x and y

xSet = data['text_clean'].values
ySet = data['Class'].values
ySet


# In[27]:


# Datatype for y is object. lets convert it into int
ySet = ySet.astype('int')
ySet


# In[ ]:





# In[28]:


xSet


# ### Splitting Train and Test Data

# In[29]:


xSet_train,xSet_test,ySet_train,ySet_test = train_test_split(xSet,ySet,test_size=0.2, random_state=10)


# In[30]:


xSet_train_CV = CV.fit_transform(xSet_train)
xSet_train_CV


# ### Training a model
# 
# With messages represented as vectors, we can finally train our spam/ham classifier. Now we can actually use almost any sort of classification algorithms. For a variety of reasons, the Naive Bayes classifier algorithm is a good choice.

# In[31]:


# Initialising the model
NB = MultinomialNB()


# In[32]:


# feed data to the model
NB.fit(xSet_train_CV,ySet_train)


# In[33]:


# Let's test CV on our test data
xSet_test_CV = CV.transform(xSet_test)


# In[34]:


# prediction for xSet_test_CV

ySet_predict = NB.predict(xSet_test_CV)
ySet_predict


# In[35]:


# Checking accuracy

accuracyScore = accuracy_score(ySet_test,ySet_predict)*100

print("Prediction Accuracy :",accuracyScore)


# ### SpamClassificationApplication

# In[36]:


msg = input("Enter Message: ") # to get the input message
msgInput = CV.transform([msg]) # 
predict = NB.predict(msgInput)
if(predict[0]==0):
    print("------------------------MESSAGE-SENT-[CHECK-SPAM-FOLDER]---------------------------")
else:
    print("---------------------------MESSAGE-SENT-[CHECK-INBOX]------------------------------")


# ## BAG OF WORDS

# We cannot pass text directly to train our models in Natural Language Processing, thus we need to convert it into numbers, which machine can understand and can perform the required modelling on it

# In[37]:


# Let's understand it with a simple example


# In[38]:


# creating a list of sentences
documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]

# Changing the text to lower case and remove the full stop from text
processed_docs = [doc.lower().replace(".","") for doc in documents]
processed_docs[3]


# In[ ]:





# In[39]:


# corpus is the collection of text
#look at the documents list
print("Our corpus: ", processed_docs)


# Initialise the object for CountVectorizer
count_vect = CountVectorizer()

#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)

#see the BOW rep for first 2 documents
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
print("BoW representation for 'man bites dog: ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["dog and dog are friends"])
print("Bow representation for 'dog and dog are friends':", temp.toarray())


# ## TF-IDF

# In **BOW approach** we saw so far, all the words in the text are treated equally important. There is no notion of some words in the document being more important than others. TF-IDF addresses this issue. It aims to quantify the importance of a given word relative to other words in the document and in the 
# 
# 
# <font color=darkviolet>  **Term Frequency (tf)** </font>
# TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:
# 
# TF(t) = (Number of times term 't' appears in a document) / (Total number of terms in the document).
# 
# 
# 
# <font color=darkviolet>  **Inverse Document Frequency (idf)** </font>
#               It measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:
# 
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).corpus. It was commonly used representation scheme for information retrieval systems, for extracting relevant documents from a corpus for given text query.
# 
# 
# 
# __Let's see an example:__
# 
# Consider a document containing 100 words wherein the word cat appears 3 times. 
# 
# The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. 
# 
# Now, assume we have 10 million documents and the word cat appears in one thousand of these. 
# 
# Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. 
# 
# Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12

# In[40]:


# Splitting x and y

X = data['text_clean'].values
y = data['Class'].values
y


# In[41]:


# Datatype for y is object. lets convert it into int
y = y.astype('int')
y


# In[42]:


type(X)


# In[43]:


## text preprocessing and feature vectorizer
# To extract features from a document of words, we import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


tf=TfidfVectorizer() ## object creation
X=tf.fit_transform(X) ## fitting and transforming the data into vectors


# In[44]:


X.shape


# In[47]:


## print feature names selected from the raw documents
# hhg=tf.get_feature_names()
# hhg


# In[48]:


# ## number of features created
# len(tf.get_feature_names())


# In[49]:


X


# In[50]:


## getting the feature vectors
X=X.toarray()


# In[51]:


## Creating training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=6)


# In[52]:


## Model creation
from sklearn.naive_bayes import BernoulliNB

## model object creation
nb=BernoulliNB(alpha=0.01) 

## fitting the model
nb.fit(X_train,y_train)

## getting the prediction
y_hat=nb.predict(X_test) 


# In[53]:


y_hat


# In[54]:


## Evaluating the model
from sklearn.metrics import classification_report,confusion_matrix


# In[55]:


print(classification_report(y_test,y_hat))


# In[56]:


## confusion matrix
pd.crosstab(y_test,y_hat)


# ### Pros of Naive Bayes
# 
# - Naive Bayes Algorithm is a fast, highly scalable algorithm
# - Naive Bayes can be classified for both binary classification and multi class classification. It provides different types of Naive Bayes Algorithms like GaussianNB, MultinominalNB, BernoulliNB.
# - It is simple algorithm that depends on doing a bunch of count.
# - Great choice for text classification problems. it's a popular choice for spam email classification.
# - It can be easily trained on small datasets.
# - Naive Bayes can handle misssing data, as they ignored when a probabilty is calculated for a class value.
# 

# ### Cons of Naive Bayes
# 
# - It considers all the features to be unrelated, so it cannot learn the relationship between features. This limits the applicability of this algorithm in real-world use cases.
# - Naive Bayes can learn individual featutre importance but can't determine the relationship among features. 

# ## Application of Naive Bayes
# 
# ##### Text classification / spam filtering / Sentiment analysis:
#  - Naive Bayes classifiers mostly used in text classification
#  - News article classification SPORTS, TECHNOLOGY etc.
#  - Spam or Ham: Naive Bayes is the most popular method for mail filtering
#  - Sentiment analysis focuses on identifying whether the customers think positively or negatively about a certain topic (product or service).
#  
#  
# ##### Recommendation System:
# - Naive Bayes classifier and Collabrative filtering together buids a recommendation system that uses machine learning and data mining techniques to filter unseen information and predict whether a user would like a given resource or not. 
# 
# 

# ### 3 Types of Naive Bayes in Scikit Learn
# 
# __Gaussian__
# 
# - It is used in classification and it assumes that features follow a normal distribution.
# 
# __Multinominal__
# - It is used for discrete counts. For eg., let's say we have a text cLassification problem. Here we consider Bernoulli trails which is one step further and instead of "word occuring in the document", we have "count how often word occurs in the document" you can think of it as "number of times outcome number_x is observed over n trails".
# 
# __Bernoulli__
# - The binomial model is useful if your feature vectors are binary (ie., Zeroes and One). One application would be text classification with 'bag of words' model where the 1s and 0s are "words occur in the document" and "word does not occur in the document" respectively.

import logging
import json
import numpy as np
import re
import string
from flask import render_template
from flask_wtf import Form
from wtforms import fields
from wtforms.validators import Required
from wtforms.widgets import TextArea
from sklearn.externals import joblib
from scipy import sparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random
import os.path
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
logreg = SklearnClassifier(LogisticRegression())
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import os
import numpy as np

# NLP
import nltk
# Import the list of stopwords from NLTK
from nltk.corpus import stopwords


from . import app

logger = logging.getLogger('app')




class PredictForm(Form):
    """Fields for Predict"""
    category = fields.SelectField('Category', choices=[ ('PC', 'PC'),
                                                        ('Wireless', 'Wireless'),
                                                        ('Baby', 'Baby'),	
                                                        ('Office Products', 'Office Products'),	
                                                        ('Beauty', 'Beauty'),	
                                                        ('Health & Personal Care', 'Health & Personal Care'),	
                                                        ('Toys', 'Toys'),	
                                                        ('Kitchen', 'Kitchen'), 	
                                                        ('Furniture', 'Furniture'),	
                                                        ('Electronics', 'Electronics'),	
                                                        ('Camera', 'Camera'),	
                                                        ('Sports', 'Sports') ,('Shoes','Shoes')])
    verified_Purchase = fields.SelectField('verified_Purchase', choices=[ ('Y', 'Yes'),('N', 'No'),('', 'Not Sure')])
    Rating = fields.SelectField('Rating', choices=[ ('0', '0'),('1', '1'), ('2', '2'), ('3', '3'),('4','4'),('5','5'),('','Not Sure')])													
    review = fields.TextAreaField('Review:', validators=[Required()])

    submit = fields.SubmitField('Submit')

class PredictForm2(Form):
    """Fields for Predict"""
    review = fields.TextAreaField('Review:', validators=[Required()])

    submit = fields.SubmitField('Submit')

#txt_file = os.getcwd() + "..\aws_labelled.txt"
txt_file = "/Users/payeldas/Project_4/Final_Pro/finalproject2/aws_labelled.txt"
df = pd.read_table(txt_file, sep='\t')
df.columns = ['reviews', 'sentiment']
tokenizer = RegexpTokenizer(r'\w+')
df['reviews'] = df['reviews'].apply(lambda x: x.lower())
df['reviews'] = df['reviews'].apply(lambda x: tokenizer.tokenize(x))
stop = set(stopwords.words('english'))
stop.remove("not")
df['stpd'] = df['reviews'].apply(lambda x: [item for item in x if item not in stop])
df['nstpd'] = df['reviews']
df['stpd_posr'] = df['stpd'].apply(lambda x: nltk.pos_tag(x))
df['nstpd_posr'] = df['nstpd'].apply(lambda x: nltk.pos_tag(x))
df['stpd_nposr'] = df['stpd']
df['nstpd_nposr'] = df['nstpd']
pos_keep = ["JJ", "JJR", "JJS", "NN", "NNP", "NNS", "RB", "RBR", "VB", "VBD", "VBG", "VBN", "VBZ"]
def remove_pos(full):
    redc =[]
    #pos_keep = ["JJ","JJR","JJS","NN","NNP","NNS","RB","RBR","VB","VBD","VBG","VBN","VBZ"]
    for pair in full:
        if pair[1] in pos_keep:
            redc.append(pair[0])
    return redc

df['stpd_posr'] = df['stpd_posr'].apply(lambda x: remove_pos(x))
df['nstpd_posr'] = df['nstpd_posr'].apply(lambda x: remove_pos(x))
ps = nltk.PorterStemmer()
ss = nltk.SnowballStemmer('english')
ls = nltk.LancasterStemmer()
df['nstpd_nposr_nstem'] = df['nstpd_nposr']
df['nstpd_posr_nstem'] = df['nstpd_posr']
df['stpd_nposr_nstem'] = df['stpd_nposr']
df['stpd_posr_nstem'] = df['stpd_posr']
df['nstpd_nposr_port'] = df['nstpd_nposr'].apply(lambda x: [ps.stem(y) for y in x])
df['nstpd_posr_port'] = df['nstpd_posr'].apply(lambda x: [ps.stem(y) for y in x])
df['stpd_nposr_port'] = df['stpd_nposr'].apply(lambda x: [ps.stem(y) for y in x])
df['stpd_posr_port'] = df['stpd_posr'].apply(lambda x: [ps.stem(y) for y in x])
df['nstpd_nposr_snow'] = df['nstpd_nposr'].apply(lambda x: [ss.stem(y) for y in x])
df['nstpd_posr_snow'] = df['nstpd_posr'].apply(lambda x: [ss.stem(y) for y in x])
df['stpd_nposr_snow'] = df['stpd_nposr'].apply(lambda x: [ss.stem(y) for y in x])
df['stpd_posr_snow'] = df['stpd_posr'].apply(lambda x: [ss.stem(y) for y in x])
df['nstpd_nposr_lanc'] = df['nstpd_nposr'].apply(lambda x: [ls.stem(y) for y in x])
df['nstpd_posr_lanc'] = df['nstpd_posr'].apply(lambda x: [ls.stem(y) for y in x])
df['stpd_nposr_lanc'] = df['stpd_nposr'].apply(lambda x: [ls.stem(y) for y in x])
df['stpd_posr_lanc'] = df['stpd_posr'].apply(lambda x: [ls.stem(y) for y in x])

@app.route('/')
def public_recipes():
    return render_template('welcomepage.html')

@app.route('/dataexplore')
def public_dataexplore():
    return render_template('dataexplore.html')
	
@app.route('/datajuyp')
def public_datajuyp():
    return render_template('sentiment_analysis_v1.html')
	
@app.route('/about')
def public_about():
    return render_template('about.html')
	
@app.route('/model3')
def public_modelthree():
    return render_template('model3.html')
	
@app.route('/model1')
def public_modelone():
    return render_template('model1.html')
	
@app.route('/welcome')
def public_welcome():
    return render_template('team.html')	

@app.route('/amazonreviewone', methods=('GET', 'POST'))	
def amazonreviewone():
    """Index page"""
    form = PredictForm()
    target_names = ['Negative', 'Positive']
    predicted = None
    my_proba = None
    proba = None
    review = 'Hello'
    Text = 'Hello'

    if form.validate_on_submit():
        # store the submitted values
        submitted_data = form.data
        product_Category = submitted_data['category']
        verified_Purchase = ''.join([random.choice(['Y', 'N']) for i in range(1)])
        Rating = submitted_data['Rating']
        Label = ''.join([random.choice(['0', '1']) for i in range(1)])
        review = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', submitted_data['review'])
        Text = review
        newtestData = []
        newtestData.clear()
        model_loc = "/Users/payeldas/Project_4/Final_Pro/finalproject2/newvib_model.pkl"
        #vector_loc = os.getcwd()+ "\\newvib_vector.pkl"
        #process_loc = os.getcwd()+ "\\newvib_preprocess.pkl"
        estimator = joblib.load(model_loc)
        #vector = joblib.load(vector_loc)
        #xprocess = joblib.load(process_loc)
        featureDict = {} 
        newtestData.append((toFeatureVector(Rating, verified_Purchase, product_Category, preProcess(Text)),Label))
        my_prediction = predictLabels(newtestData, estimator)
        predicted = ''.join(my_prediction)
        proba = ''.join(my_prediction)
    return render_template('amazonreviewone.html',
                       form=form,
                       prob=proba,
                       prediction=predicted)
def predictLabels(reviewSamples, classifier):
    return classifier.classify_many(map(lambda t: t[0], reviewSamples))

def toFeatureVector(Rating, verified_Purchase, product_Category, tokens):
    localDict = {}
    featureDict ={}
    
#Rating

    #print(Rating)
    featureDict["R"] = 1   
    localDict["R"] = Rating

#Verified_Purchase
  
    featureDict["VP"] = 1
            
    if verified_Purchase == "N":
        localDict["VP"] = 0
    else:
        localDict["VP"] = 1

#Product_Category

    
    if product_Category not in featureDict:
        featureDict[product_Category] = 1
    else:
        featureDict[product_Category] = +1
            
    if product_Category not in localDict:
        localDict[product_Category] = 1
    else:
        localDict[product_Category] = +1
            
            
#Text        

    for token in tokens:
        if token not in featureDict:
            featureDict[token] = 1
        else:
            featureDict[token] = +1
            
        if token not in localDict:
            localDict[token] = 1
        else:
            localDict[token] = +1
    
    return localDict

def preProcess(text):                                                                                       
    # Should return a list of tokens                                                                        
    lemmatizer = WordNetLemmatizer()                                                                        
    filtered_tokens = []                                                                                    
    lemmatized_tokens = []                                                                                  
    table = str.maketrans({key: None for key in string.punctuation})                                        
    stop_words = set(stopwords.words('english'))                                                            
    text = text.translate(table)                                                                            
    for w in text.split(" "):                                                                               
        if w not in stop_words:                                                                             
            lemmatized_tokens.append(lemmatizer.lemmatize(w.lower()))                                       
        filtered_tokens = [' '.join(l) for l in nltk.bigrams(lemmatized_tokens)] + lemmatized_tokens        
    return filtered_tokens   
	
	



# ### Function to Convert the Data into a Feature Set

# In[15]:

## Transform data into list of ([tokens],sentiment label)
def createTrainingDataNLTK(sentences,labels):
    rdata = np.vstack([sentences,labels])
    rdata = np.transpose(rdata)
    data = list();
    for i in range(0,len(rdata)):
        tokens = rdata[i][0].split(" ")
        d_tuple = (tokens, rdata[i][1]);
        data.append(d_tuple)
    return data;


# ## Create the Training Data

# In[16]:

# merge the words into sentence to use current implementation of createTrainingData
def create_nltk_train_data (feature_reduction):
    df['sentences'] = df[feature_reduction].apply(lambda x: " ".join(x))
    x_label = "sentences"
    y_label = "sentiment"
    nltk_train_data = createTrainingDataNLTK(df[x_label],df[y_label])
    return nltk_train_data


# ## Functions to Train the Classifers and Run it Against Test Data

# In[17]:

# returns the accuracy of the test data for Naive Bayes
# when predicted against fitted training data
def train_nb(training_set):
    
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_set])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    
    training_set = sentim_analyzer.apply_features(training_set)
                                              
    trainer = NaiveBayesClassifier.train
    classifier = sentim_analyzer.train(trainer, training_set)                         
    return [sentim_analyzer,classifier]


# In[18]:

# returns the accuracy of the test data for Logistic Regression
# when predicted against fitted training data
def train_lr(training_set):
    
    sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_set])
    unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
    sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
    
    training_set = sentim_analyzer.apply_features(training_set)
                                              
    trainer = logreg.train
    classifier = sentim_analyzer.train(trainer, training_set)                 
    return [sentim_analyzer,classifier]


# In[19]:
def predict_lr(text, trained):
    tok =  tokenizer.tokenize(text)
    tok = nltk.pos_tag(tok)
    tok = remove_pos(tok)
    fs = trained[0].apply_features([(tok)])
    return trained[1].prob_classify(fs[0][0])

# In[20]:
def predict_nb(text, trained):
    tok =  tokenizer.tokenize(text)
    tok = nltk.pos_tag(tok)
    tok = remove_pos(tok)
    fs = trained[0].apply_features([(tok)])
    return trained[1].prob_classify(fs[0][0])

# Create the Flask Restful API endpoints to expose the trained models







def count_unique(words):
    uniq = set()
    for sentence in words:
        for word in sentence:
            uniq.add(word)
    return len(uniq)

@app.route('/datacompare', methods=('GET', 'POST'))	
def predict1():
    form = PredictForm2()
    score1=0
    score2=0
    if form.validate_on_submit():
        submitted_data = form.data
        sentence = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', submitted_data['review'])
        ds_l = train_lr(create_nltk_train_data('nstpd_nposr_snow'))
        ds_n = train_nb(create_nltk_train_data('nstpd_nposr_nstem'))
        dist1 = predict_lr(sentence, ds_l)
        score1 = str(round(100 * dist1.prob(1)))
        dist2 = predict_nb(sentence, ds_n)
        score2 = str(round(100 * dist2.prob(1)))
    return render_template('datacompare.html',
                       form=form,
                       prob0=score1,
                       prob1=score2)
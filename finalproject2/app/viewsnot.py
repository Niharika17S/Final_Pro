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

@app.route('/')
def public_recipes():
    return render_template('welcomepage.html')


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
        verified_Purchase = submitted_data['verified_Purchase']
        Rating = submitted_data['Rating']
        Label = ''.join([random.choice(['0', '1']) for i in range(1)])
        review = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])').sub(r' \1 ', submitted_data['review'])
        Text = review
        newtestData = []
        newtestData.clear()
        model_loc = os.getcwd() + "\\newvib_model.pkl"
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
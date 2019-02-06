# Project-3|| Amazon Review Analysis using Machine Learning.

![image](Images/amazon-logo.jpg)
The goal of this porject is text classification that is, to automatically classify the text documents into one or more defined categories. Some examples of text classification are:
* Understanding audience sentiment from social media,
* Detection of spam and non-spam emails.

Text Classification is an example of supervised machine learning task since a labelled dataset containing text documents and their labels is used for train a classifier. An end-to-end text classification pipeline is composed of three main components:

1. Dataset Preparation: The first step is the Dataset Preparation step which includes the process of loading a dataset and performing basic pre-processing. The dataset is then splitted into train and validation sets.
2. Feature Engineering: The next step is the Feature Engineering in which the raw dataset is transformed into flat features which can be used in a machine learning model. This step also includes the process of creating new features from the existing data.
3. Model Training: The final step is the Model Building step in which a machine learning model is trained on a labelled dataset.
In this project we use have used Supervised Machine Learning Algoriths :
Naïve Bayes  (NB), 
Support  Vector  Machine  (SVM),
TextBlob,
Logistics Regression for classification of the reivews.

Data Source:
![image](https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt)
=========================
## Sentiment Analysis:
Opinion Mining (OM), also known as Sentiment Analysis (SA), is a  common text categorization task which involves extraction of sentiment, the positive or negative orientation that a writer expresses toward some object. 

Sentiment analysis in Machine Learning is a process of automatically identifying whether a user-generated text expresses positive, negative or neutral opinion about an entity (i.e. product, people, topic, event etc).
The sentiment is usually formulated as a two-class  classification  problem,  positive  and  negative.
This analysis can be determined 
1)Document level(used Logistic Regression and Nayes Bayes Analyzer) 
2)Sentence level(used Textblob)
The document level aims to classify an opinion document as a negative or positive opinion.
The sentence level using SA aims to setup opinion stated in every sentence. 

## Methodology:

* Getting your machine ready
* Dataset preparation
* Preprocesing
* Feature Extraction
* Model Preparation
* Evaluating and Comparing the models
* Summary.






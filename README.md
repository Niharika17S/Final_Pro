# Project-3|| Amazon Review Analysis using Machine Learning.
============================================================


The goal of this project is text classification that is, to automatically classify the text documents into one or more defined categories. Some examples of text classification are:
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

![image](https://user-images.githubusercontent.com/41707119/52385242-b2609300-2a4e-11e9-86c0-cfdc5ccf073a.png)
### Scope of the Analysis :

1. What product category is widely rated.
2. What is the correlation between “Verified” reviews and ratings 
3. Identifying fake Reviews?
4. Identify the sentiment ( positive or negative ) of each review

### Data Source and Content :

Source :  https://www.kaggle.com/lievgarcia/amazon-reviews

Content : 
Product_id             : The unique Product ID the review pertains to
Product CATEGORY : Broad product category that can be used to group reviews 
Product Title         : Title of the product
Review title            : The HEADLINE of the review
Review text             :  the body of the text 
Rating                      : The 1-5 star rating of the review.
Verified                   : The review is on a verified purchase.

STEPs:
-------
* Exploratory Data Analysis
* Tokenize & POS Tagging/Remove stopwords
* Testing and Evaluate with Logistic Regression , Naive Bayes, SVM(support vector machine), K-Folds
* Train for accuracy
* Flask App for '.pkl' model 

Data exploration and graphs to understand:
------------------------------------------

Captured a small part of the dataset on which the model has been created:
![xnip2019-02-06_11-42-55](https://user-images.githubusercontent.com/41707119/52384818-04a0b480-2a4d-11e9-9b15-0e9c1ccb64cd.jpg)
![xnip2019-02-06_02-32-25](https://user-images.githubusercontent.com/41707119/52384805-04081e00-2a4d-11e9-8234-f70b41a992c8.jpg)
![xnip2019-02-06_02-32-53](https://user-images.githubusercontent.com/41707119/52384806-04081e00-2a4d-11e9-9702-d8c60e86ce22.jpg)
![xnip2019-02-06_02-33-22](https://user-images.githubusercontent.com/41707119/52384807-04081e00-2a4d-11e9-9fb3-b8e6503445d4.jpg)
![xnip2019-02-06_02-34-00](https://user-images.githubusercontent.com/41707119/52384808-04081e00-2a4d-11e9-8b42-d43b3d4e98e2.jpg)
![xnip2019-02-06_02-34-49](https://user-images.githubusercontent.com/41707119/52384809-04081e00-2a4d-11e9-84b4-6f0352e58e48.jpg)
![xnip2019-02-06_02-37-41](https://user-images.githubusercontent.com/41707119/52384811-04081e00-2a4d-11e9-879e-1977bde0ac13.jpg)
![xnip2019-02-06_02-38-35](https://user-images.githubusercontent.com/41707119/52384812-04081e00-2a4d-11e9-911a-0f86ca6430e7.jpg)
![xnip2019-02-06_02-39-27](https://user-images.githubusercontent.com/41707119/52384813-04a0b480-2a4d-11e9-8702-303f6f172bc2.jpg)
![xnip2019-02-06_02-42-19](https://user-images.githubusercontent.com/41707119/52384814-04a0b480-2a4d-11e9-9ae9-cb790cc65a28.jpg)
![xnip2019-02-06_03-09-40](https://user-images.githubusercontent.com/41707119/52384815-04a0b480-2a4d-11e9-9edd-a3110cce8f97.jpg)
![xnip2019-02-06_03-10-56](https://user-images.githubusercontent.com/41707119/52384816-04a0b480-2a4d-11e9-9959-a3dd2395ab83.jpg)
![xnip2019-02-06_03-12-28](https://user-images.githubusercontent.com/41707119/52384817-04a0b480-2a4d-11e9-9f24-152a7aab5e25.jpg)


Comparative view of two Model (SVM/Naive Bayes):
------------------------------------------------
![image](https://user-images.githubusercontent.com/41707119/52385094-23ec1180-2a4e-11e9-9d09-b6468624c372.png)
![image](https://user-images.githubusercontent.com/41707119/52385160-71687e80-2a4e-11e9-9458-476adef98869.png)

Sentiment Analysis:
-------------------
![image](https://user-images.githubusercontent.com/41707119/52385358-3286f880-2a4f-11e9-87ac-a5794d4f9d1c.png)
![image](https://user-images.githubusercontent.com/41707119/52385411-6c57ff00-2a4f-11e9-8775-2b007febedde.png)


## Methodology:

* Getting your machine ready
* Dataset preparation
* Preprocesing
* Feature Extraction
* Model Preparation
* Evaluating and Comparing the models
* Summary.






# Final_Pro|| Analysis On Amazon Reviews:
==============================================
### Problem Solving using Machine Learning

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

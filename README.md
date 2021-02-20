
### Abstract 

*Sentiment analysis, the extraction of underlying meaning of reviews of positive and negative is becoming one of the most attractive tools for researchers and businesses to group customer feedback and better determine their desires and needs. Recommender systems are also becoming more and more important in the marketing industry, to streamline targeted advertising. Natural language processing, machine learning and neural network approaches on large volumes of text, make it possible to extract sentiment analysis and build recommender systems. In this project we will build these two tools based on European hotel reviews.*

*The main goal is using neural network models for sentiment analysis on the text data set which includes reviews of Hotels, and comparing performance with other classification algorithms. We also would like to explore how to use this result as a feature in different applications such as recommender systems.*


### Introduction

Sentiment analysis is the automated process of understanding the sentiment or opinion of a given text. It is one of the most common classification algorithms. The sentimental tool analyzes the reviews with Natural Language Processing, Machine Learning and Neural Network approaches to evaluate whether the underlying sentiment of a small text is positive, negative or neutral. This is usually some form of customer feedback or review, for example: customer service, an item on an ecommerce website, or, as used in this project hotel reviews.
The goal is to apply different approaches (train different models) on a sample of European Hotel Reviews to determine the sentiment of each review. We will then compare the results and metrics of each model’s attempt at determining the sentiment of an unseen review (testing) and determine the best performing algorithm. These reviews, and most reviews in general, are technically labelled, as the user will likely be asked to include a quantifiable review (out of 5 stars). However, the value added by these experiments is the learning done by the data models, so that they may quantify unlabelled data, or even follow up with customers whose ratings don’t match the sentiment expressed in their review text. Determining which model performs best will offer better results with respect to these goals. 
The secondary goal is to use the result of sentimental analysis, along with some other features that will be extracted, is to create a recommender system. The recommender system provides hotels that are more relevant to the search item or are close to the search history of the user and analyzed preferences of the user. The aim of the recommender system is to improve the quality of search results and recommend certain hotels that are more likely to interest certain users.


### Materials and Methods

The dataset is composed of 35,000 data, pulled from [Kaggle](https://www.kaggle.com/datafiniti/hotel-reviews). Each data instance (row) corresponds to one review of a hotel in Europe. Of the nineteen columns in the table, some provided features include hotel location, name, rating, review text, username, etc. A random sample of 10,000 will be selected, Training and testing data will be randomly shuffled and then split into 80/20, respectively. So we will have 800 samples for training and 200 for testing the models. 

Example of Row:

![example-data](assets/data-example.png)


#### Sentiment Analysis
For sentiment analysis, we will mostly be interested in features: review.text and review.date, review.rating. The first step is to analyze review.text to prepare for pre processing and feature engineering. Given review.text is text data, we need to recognize what information (words) included in this text is useful for NLP approaches and what is not. After extracting useful information from the review.text, we apply some feature engineering on reviews.rating and review.date to create more valuable features. We apply simple linear regression to the dataset  and then by calculating least square error we figured that we have a linear data set.  As a result of that we consider applying multiple Machine Learning and Neural Network algorithms on our dataset.


Mostly we are going to work on the review column and apply some text preprocessing to clean the data. We assume that data requires lots of preprocessing since the reviews are written by users and they include lots of unnecessary stopwords and parts of speech which are not useful for feeding into text classification models. The reviews are multilingual, we will focus on English reviews, and therefore implement some filtering with respect to the language used. Since the data is text, we also need to do some vectorization to convert it to numeric data to be able to use it for training models. Also, the data set is unlabeled so we need to do some feature engineering to create labels for example by using the rating column. 

#### Recommender System
For the recommender system, because we want to try all the different approaches such as democratic and collaborative filtering, and content based filtering. 
Content-based filtering will recommend a similar hotel based on a particular review. The idea behind this system is that if a person liked a hotel and gave a positive review, they will also like a hotel with a similar review.We will calculate pairwise similarity scores for all hotels based on their reviews and recommend hotels based on that similarity score. We will use Term Frequency-Inverse Document Frequency to transform the reviews into vectors.
Demographic filtering gives  a  generalized  recommenda-tions to every user,  based on popularity of the hotel. We will need a metric to calculate the score for each hotel, then we sort  hotels based on their scores and return the best rated ones as the recommendation. We will use the IMDb’s rating system formula to calculate the score for each hotel.




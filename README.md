# Project Title
### Disaster Response Pipeline Project

# Table of contents
* Overview
* Components
* Files
* Technologies
  
# Overview
This project aims at creating an ETL pipeline that takes messages received from people during disasters, cleans the messages text and then uses a machine learning model to classify the category of the emergency text. We have 36 main emergency categories. We will update our work on a flask web application that will have a text box to enter the message, and then it retrieves the categories related to the message. This will help emergency workers to easily identify people needs during crisis and assign it to the concerned organizations.

# Components
Our project consists of 3 main components:

1- ETL pipeline:
In this part, We will read a labelled dataset that will be used to build the classifier, clean the data, and then store it in a SQLite database. 

2- Machine Learning:
We will create a machine learning pipeline that will be trained on our data to classify emergency messages into 36 categories (multi-output classification). train_classifier.py.

3- Flask App:
We will display the results in a Flask web app that has a text box in which we can enter the emergency messages, and see the classification categories. The app also displays visualizations for the top 10 and lowest 10 categories.

# Files:

Project Directory

- app
| - template

| |- master.html  # main page of web app

| |- go.html  # classification result page of web app

# Technologies
This project uses Python 3.10.9 version


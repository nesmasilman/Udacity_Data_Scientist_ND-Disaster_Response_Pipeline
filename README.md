# Project Title
### Disaster Response Pipeline Project

# Table of contents
* Overview
* Components
* Files
* Technologies
* Outcome
* Resources

# Overview
This project aims at creating an ETL pipeline that takes messages received from people during disasters, cleans the messages text and then uses a machine learning model to classify the category of the emergency text. We have 36 main emergency categories. We will update our work on a flask web application that will have a text box to enter the message, and then it retrieves the categories related to the message. This will help emergency workers to easily identify people needs during crisis and assign it to the concerned organizations.

# Components
Our project consists of 3 main components:

1- ETL pipeline
In this part, We will read a labelled dataset that will be used to build the classifier, clean the data, and then store it in a SQLite database. process_data.py.

2- Machine Learning
We will create a machine learning pipeline that will be trained on our data to classify emergency messages into 36 categories (multi-output classification). train_classifier.py.

3- Flask App
We will display the results in a Flask web app that has a text box in which we can enter the emergency messages, and see the classification categories. The app also displays visualizations for the top 10 and lowest 10 categories.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

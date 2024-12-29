# Project Title
### Disaster Response Pipeline Project

# Table of contents
* Overview
* Datasets
* Technologies
* Outcome
* Resources

# Overview
This project aims at creating an ETL pipeline that takes messages received from people during disasters, cleans the messages text and then uses a machine learning model to classify the category of the emergency text. We have 36 main emergency categories. We will update our work on a flask web application that will have a text box to enter the message, and then it retrieves the categories related to the message. This will help emergency workers to easily access people during disasters.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

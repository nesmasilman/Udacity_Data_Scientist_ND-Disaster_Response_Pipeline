import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.corpus import stopwords
import re
from sklearn.multioutput import MultiOutputClassifier
import pickle

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

def load_data(database_filepath):
    """Reads the data from filepath, splits into independant columns, 
        target column and categories names"""
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('df', engine)

    # Drop un-needed column
    df1 = df.drop(columns=['original'], axis=1) 

    # Drop null values
    df2 = df1.dropna(axis=0) 

    # Define target and independant variables
    X = df2['message']
    Y = df2.drop(columns=['id', 'message', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

def tokenize(text):
    """Takes in a text, performs all pre-processing steps, and return a list of tokens after pre-processing"""

    tokens = word_tokenize(text) # Tokenize words
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()       #Normalize tokens
        stemmed = PorterStemmer().stem(clean_tok) # Stem tokens
        clean_tokens.append(stemmed) # Add to an empty list
    words = [w for w in clean_tokens if w not in stopwords.words("english")] # Remove stop words

    return words

def build_model():
    """Performs Gridsearch on pipeline components"""
    # Create a pipeline using knn classifier
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    # Define Gridsearch parameters grid
    parameters = {'clf__estimator__n_estimators': [20, 30]
    }
    # Perform GridSearchCV
    cv = GridSearchCV(pipeline, param_grid= parameters, cv=2
                      , verbose=3, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Retrieves model predictions and classification reports for each column"""
    # Retrieve predictions on the test set
    y_predicted = model.predict(X_test)

    # Create a dataframe from predictions
    y_pred = pd.DataFrame(y_predicted, columns=category_names)
    
    # Iterate on all columns and retrieve classification report
    classification_scores = []
    for col in category_names:
        score = classification_report(Y_test[col], y_pred[col])
        classification_scores.append(score)
  
        return classification_scores

def save_model(model, model_filepath):
    """Saves the trained model as a pickle file in the given path"""
    
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
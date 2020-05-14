import sys
import re
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score
from scipy.stats import hmean
from scipy.stats.mstats import gmean

nltk.download(['punkt', 'wordnet','stopwords'])
nltk.download(['averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
        - Load cleaned data from database into dataframe
        - Extract X, Y data
        
        Args:
            database_filepath (str): File path of database
            
        Returns:
            X (pandas dataframe): Contains messages
            Y (pandas dataframe): Contains categories of disaster
            category_names (list): Name of categories
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    #engine.dispose()
    df = pd.read_sql_table("Disaster_Response", engine)
    # Extract X and Y
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
        - Tokenize text
        
        Args:
            text (str): Message string
            
        Returns:
            tokens (list): List of tokens
    """
    #converting all the text to lowercase
    text.lower()
    # removing punctuation characters from the text and replacing them with an empty space
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # use the word tokenizer to convert text into tokens
    tokens = word_tokenize(text)
    # removing stopwords by calling a "for loop" for the tokens and using english stopwords from nltk!
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    # Using lemmatization to strip all the words
    tokens = [WordNetLemmatizer().lemmatize(word).strip() for word in tokens]
    # finally return the tokens
    return tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def model_pipeline():
    """
        - Build pipeline for creating model
        
        Returns:
            pipeline_modified: Pipeline for creating the model
    """
    # Build pipeline
    pipeline_modified = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),

                ('starting_verb', StartingVerbExtractor())
            ])),

            ('clf', RandomForestClassifier())
        ])
    return pipeline_modified



def build_model():
    """
        - Build the model with the proposed parameters
        
        Returns:
            cv (GridSearchCV): Machine learning model
    """
    model = model_pipeline()
    # Parameter selection using GridSearchCV
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }
    cv = GridSearchCV(model, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        - Evaluate model
        
        Args:
            model (Predicter model): Machine learning model (Predicter)
            X_test (pandas series): Test data set of X
            Y_test (pandas dataframe): Test data set of Y
            category_names (list): Name of categories
    """
    # Predict test data
    predict_y = model.predict(X_test)
    
    # Evaluate
    for i in range(len(category_names)):
        category = category_names[i]
        print(category)
        print(classification_report(Y_test[category], predict_y[:, i]))


def save_model(model, model_filepath):
    """
        - Saves model in pickle file
        
        Args:
            model (Predicter model): Machine learning model (Predicter)
            model_filepath (str): Path where model will save.
    """
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


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

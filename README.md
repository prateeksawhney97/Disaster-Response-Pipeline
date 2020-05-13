# Disaster Response Pipeline Project

## Udacity Data Scientist Nanodegree

### Description:

This Project is a part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disasters. The aim of this project is to build a Natural Language Processing tool that categorize messages.

The Project is divided into the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure.
2. Machine Learning Pipeline to train a model which is able to classify text messages in 36 categories.
3. Web Application using Flask to show model results and predictions in real time.

### Installing:

Clone this repository

```
git clone git@github.com:prateeksawhney97/Disaster-Response-Pipeline.git
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster_Response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Screenshots:

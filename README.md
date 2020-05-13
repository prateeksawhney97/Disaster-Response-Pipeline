# Disaster Response Pipeline Project

## Udacity Data Scientist Nanodegree

### Description:

This Project is a part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disasters. The aim of this project is to build a Natural Language Processing tool that categorize messages.

The Project is divided into the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure.
2. Machine Learning Pipeline to train a model which is able to classify text messages in 36 categories.
3. Web Application using Flask to show model results and predictions in real time.

### Data:

The data in this project comes from Figure Eight - Multilingual Disaster Response Messages. This dataset contains 30,000 messages drawn from events including an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, super-storm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters.

The data has been encoded with 36 different categories related to disaster response and has been stripped of messages with sensitive information in their entirety.

Data includes 2 csv files:

1. disaster_messages.csv: Messages data.
2. disaster_categories.csv: Disaster categories of messages.

### Folder Structure:

* app
    * | - templates
        * |- master.html # main page of web application
        * |- go.html # classification result page of web application
    * |- run.py # Flask file that runs application

* data
   * |- disaster_categories.csv # data to process
   * |- ML Pipeline Preparation.ipynb
   * |- ETL Pipeline Preparation.ipynb
   * |- disaster_messages.csv # data to process
   * |- process_data.py
   * |- Disaster_Response.db # database to save clean data to
   
* models
   * |- train_classifier.py
   * |- classifier.pkl # saved model

* README.md

### Installation:

This project requires Python 3.x and the following Python libraries:

* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

### Instructions:

1. Clone this repository

```
git clone git@github.com:prateeksawhney97/Disaster-Response-Pipeline.git
```

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster_Response.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

### Screenshots:

##### 1. Home Page

![Screenshot from 2020-05-12 22-57-43](https://user-images.githubusercontent.com/34116562/81772729-39c76c80-9504-11ea-85e9-e5f45db2cbf6.png)
![Screenshot from 2020-05-12 22-57-48](https://user-images.githubusercontent.com/34116562/81772741-3d5af380-9504-11ea-9ce3-afc7e78dddda.png)
![Screenshot from 2020-05-12 22-57-53](https://user-images.githubusercontent.com/34116562/81772747-41871100-9504-11ea-942b-38846c7d49bf.png)
![Screenshot from 2020-05-12 22-58-21](https://user-images.githubusercontent.com/34116562/81772751-451a9800-9504-11ea-9fe6-76bae5341ad4.png)

##### 2. Classify Messages Page

![Screenshot from 2020-05-12 22-58-46](https://user-images.githubusercontent.com/34116562/81772761-4c41a600-9504-11ea-88af-90ecd9eb4ff5.png)

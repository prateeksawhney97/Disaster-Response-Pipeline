import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
        - Loads the data from csv files into dataframe
        - Merge the 2 dataframe
        
        Args:
            messages_filepath (str): File path of message
            categories_filepath (str): File pathe of category
            
        Returns:
            pandas dataframe: Merged data that is loaded
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    """
        - Cleans the data
        
        Args:
            df (pandas dataframe): Merged dataframe
            
        Returns:
            pandas dataframe: Cleaned and structured dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda name: name[:-2]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df = df.drop(["categories"], axis=1)
    # concatenates the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop the duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
        - Save clean data into sqlite database
        
        Args:
            df (pandas dataframe): Cleaned data
            database_filename (str): Database name to save
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disaster_Response', engine, index=False) 
    engine.dispose()


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

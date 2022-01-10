import sys
import pandas as pd 
from sqlalchemy import create_engine
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()


def load_data(messages_filepath, categories_filepath):
    '''
    loads data from user inputted location
    input: 2 str locations
    output: dataframe consisting of both datasets
    '''
    # read in message
    messages = pd.read_csv(messages_filepath, sep = ',', index_col = 'id')
    # read in categories
    categories = pd.read_csv(categories_filepath, sep = ',', index_col = 'id')
    # concat data together
    df = messages.join(categories)
    return df


def clean_data(df):
    '''
    cleans dataframe
    input: dataframe
    output: cleaned dataframe
    '''

    # split data into separate columns
    categories = df['categories'].str.split(';',expand = True)
    # get list of labels
    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0])
    # set columns names as labels
    categories.columns = category_colnames
    # convert label data into 1 or 0
    categories = categories.applymap(lambda x:int(x.split('-')[1]))
    # replace 2 with 0 - according to data page
    categories = categories.replace(2,0)
    # delete old categories column
    del df['categories']
    # join newly made label data with original data
    df = df.join(categories)
    # remove duplicatees
    df = df[~df.duplicated()]
    # apply vader sentiment analysis
    df['VADER'] = df['message'].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df


def save_data(df, database_filename):
    '''
    saves data to a sqlite database
    input: df to save and filepath
    output: None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_tweets', engine, index=False)  
    pass


def main():
    '''
    runs the etl functions
    '''
    
    # makes sure the user gives 3 inputs
    if len(sys.argv) == 4:

        # save inputs as variables
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]


        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        # save concatenated data
        df = load_data(messages_filepath, categories_filepath)

        # clean data
        print('Cleaning data...')
        df = clean_data(df)
        
        # save into sqlite
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    

    # print error message in case there is not enough inputs
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
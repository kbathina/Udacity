import sys
import pandas as pd 
from sqlalchemy import create_engine
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath, sep = ',', index_col = 'id')
    categories = pd.read_csv(categories_filepath, sep = ',', index_col = 'id')
    df = messages.join(categories)
    return df


def clean_data(df):
    categories = df['categories'].str.split(';',expand = True)
    category_colnames = categories.iloc[0].apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    categories = categories.applymap(lambda x:int(x.split('-')[1])).replace(2,0)
    del df['categories']
    df = df.join(categories)
    df = df[~df.duplicated()]
    df['VADER'] = df['message'].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_tweets', engine, index=False)  


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
from sqlalchemy import create_engine
import pandas as pd
import re
import numpy as np
import pickle

import nltk
from nltk import TweetTokenizer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
tt = TweetTokenizer(preserve_case = False, strip_handles = True)
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_tweets', con = engine)

    X = df['message'].values
    Y = df[df.columns[3:]].values

    return X,Y

def tokenize(text):
    text = tt.tokenize(text)
    text = [word for word in text if word not in stopwords]
    text = [punc_stripper(word) for word in text]
    text = [word for word in text if word]
    text = [lemmatizer.lemmatize(word) for word in text]
    
    return text


def build_model():
    pipeline = Pipeline([
        ('count',CountVectorizer(tokenizer=tokenize)),
        ('clf',MultiOutputClassifier(MultinomialNB()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    return model.transform(X_test,Y_test, category_names)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


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
from sqlalchemy import create_engine
import pandas as pd
import re
import numpy as np
import pickle
from collections import Counter
import sys

import nltk
from nltk import TweetTokenizer,word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
# list of tags from nltk
pos_tags = ['CC','CD','DT','EX','FW','IN','JJ','JJR',
 'JJS','LS','MD','NN','NNP','NNPS','NNS',
 'PDT','POS','PRP','PRP$','RB','RBR','RBS',
 'RP','SYM','TO','UH','VB','VBD','VBG','VBN',
 'VBP','VBZ','WDT','WP','WP$','WRB']


from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# initialize tweet tokenizer - remove usernames and lowercase
tt = TweetTokenizer(preserve_case = False, strip_handles = True)

# initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    '''
    reads in data from sqlite database
    input: filpath to database
    output: messages, labels, and column names
    '''
    # open connection to database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # read tweets from table into df
    df = pd.read_sql_table('disaster_tweets', con = engine)

    # store tweet message
    X = df['message'].values
    # drop non-label columns
    to_drop = ['message','original','genre', 'VADER']
    # store label column names
    columns = df.drop(to_drop, axis=1).columns
    # store labels
    Y = df.drop(to_drop, axis=1).values

    # return tweet, labels, and columns
    return X,Y, columns


def punc_stripper(word):
    '''
    return string if it is not punctuation except for emojis
    input: string
    output: input if it is 
    '''
    #  return the word if the length is greater than 1
    if len(word) > 1: 
        return word
    # return the word if the length is 1 and it is alphanumeric
    if len(word) == 1 and word.isalnum():
        return word

def tokenize(text):
    '''
    custom tokenizer for feature engineering
    input: string of text
    output: cleaned string
    '''
    # first tokenize using nltk tweet tokenizer
    text = tt.tokenize(text)
    # remove stopwords
    text = [word for word in text if word not in stopwords]
    # remove punctuation - the tweet tokenizer will keep separate punctuation into separate tokens except for emojis
    text = [punc_stripper(word) for word in text]
    # remove None from list of tokens
    text = [word for word in text if word]
    # lemmatize remaining tokens
    text = [lemmatizer.lemmatize(word) for word in text]
    
    # return ordered list of tokens
    return text

class POS_Counter(BaseEstimator, TransformerMixin):
    '''
    class that counts parts of speech tags using the nltk pos
    '''

    def pos_dict_maker(self):
        '''
        initializes a count dictionary with all possible pos tags from nltk
        input: None
        output: count dict
        '''
        # initialize empty dictionary
        pos_dict = {}

        # add each tag to the dictionary with a count of 0
        for pos in pos_tags:
            pos_dict[pos] = 0
        return pos_dict

    def fit(self, X, y=None):
        '''
        empty fit function
        '''
        return self

    def transform(self, X):
        '''
        builds parts of speech count dictionar as features for ML
        input: collection of text (str)
        output: df of parts of speech counts
        '''
        # intitialize list to hold count dictionaries for each text
        X_tagged = []

        # iterate through each text
        for text in X:
            # initialize a count dictionary
            pos_dict = self.pos_dict_maker()
            
            # count number of each parts of speech using nltk pos_tag
            counts = Counter([pos for word,pos in pos_tag(word_tokenize(text))])     
            # update count dictionary       
            for k,v in counts.items():
                if k.isalnum():
                    pos_dict[k] = v
            # append to list
            X_tagged.append(pos_dict)

        # convert to pandas and replace NA with 0 and then return
        return pd.DataFrame(X_tagged).fillna(0)


def build_model():
    '''
    builds Ml pipeline
    input: nothing
    output: sklearn model
    '''
    
    # tfidf features to be used in machine learning
    # calculates tfidf values and then standadrizes them
    # used in 'features' FeatureUnion
    tfidf_scaled = Pipeline([
        ('tfidf',TfidfVectorizer(tokenizer=tokenize)),
        ('scalar',StandardScaler(with_mean=False))
    ])

    # FeatureUnion for 2 sets of features
    # 1. POS counts 
    # 2. tfidf from above
    features = FeatureUnion([
        ('pos_counter',POS_Counter()),
        ('tfidf_scaled',tfidf_scaled)
    ])

    # full ML pipeline
    # builds features
    # fit using a multinomial naive bayes algorithm that is extended for multi label classification
    pipeline = Pipeline([
        ('features',features),
        ('clf',MultiOutputClassifier(MultinomialNB()))
    ])

    # set of parameters for gridsearch
    parameters = {
        # smoothing parameter for the naive bayes algorithm
        'clf__estimator__alpha': [0.001,0.5,1],
        # binary or raw counts for tfidf
        'features__tfidf_scaled__tfidf__binary': [True,False],
        # min frequency for word to be added to tfidf dictionary
        'features__tfidf_scaled__tfidf__min_df': [5,50]
    }

    # build model with a gridsearch]
    model = GridSearchCV(estimator=pipeline,
        param_grid=parameters,verbose = 2,n_jobs = -1)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluation f1, precision, and recall of the model
    input: the model, test data, test labels, and names of features
    output: classification_report from sklearn for each feature
    '''

    # predict labels for test data using model
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test,y_pred,target_names = category_names))
    pass

def save_model(model, model_filepath):
    '''
    save model to the user defined location
    input: model and location (str)
    output: None
    '''
    # pickle serialize model and store to user defined location
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    # user must give two inputs for the function
    # else print error statement
    if len(sys.argv) == 3:
        # save the inputs
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        # load tweets, labels, and label names
        X, Y, category_names = load_data(database_filepath)
        # split data into test/train data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        # build and output model
        model = build_model()
        
        print('Training model...')
        # fit to training data
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        #print model evaluation
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        # save model to user defined location
        save_model(model, model_filepath)

        print('Trained model saved!')

    # printed error statement if not enough inputs from user
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
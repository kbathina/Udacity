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
tt = TweetTokenizer(preserve_case = False, strip_handles = True)
lemmatizer = WordNetLemmatizer()
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_tweets', con = engine)

    X = df['message'].values
    columns = df[df.columns[3:]].columns
    Y = df[df.columns[3:]].values

    return X,Y, columns

def punc_stripper(word):
    if len(word) > 1: 
        return word
    if len(word) == 1 and word.isalnum():
        return word

def tokenize(text):
    text = tt.tokenize(text)
    text = [word for word in text if word not in stopwords]
    text = [punc_stripper(word) for word in text]
    text = [word for word in text if word]
    text = [lemmatizer.lemmatize(word) for word in text]
    
    return text

class POS_Counter(BaseEstimator, TransformerMixin):

    def pos_dict_maker(self):
        pos_dict = {}
        for pos in pos_tags:
            pos_dict[pos] = 0
        return pos_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = []
        for text in X:
            pos_dict = self.pos_dict_maker()
            counts = Counter([pos for word,pos in pos_tag(word_tokenize(text))])            
            for k,v in counts.items():
                if k.isalnum():
                    pos_dict[k] = v
                    
            X_tagged.append(pos_dict)

        return pd.DataFrame(X_tagged).fillna(0)


def build_model():
    tfidf_scaled = Pipeline([
        ('tfidf',TfidfVectorizer(tokenizer=tokenize)),
        ('scalar',StandardScaler(with_mean=False))
    ])

    features = FeatureUnion([
        ('pos_counter',POS_Counter()),
        ('tfidf_scaled',tfidf_scaled)
    ])

    pipeline = Pipeline([
        ('features',features),
        ('clf',MultiOutputClassifier(MultinomialNB()))
    ])

    parameters = {
        'clf__estimator__alpha': [0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
        'features__tfidf_scaled__tfidf__binary': [True,False],
        'features__tfidf_scaled__tfidf__min_df': [1,5,10,50]
    }

    model = GridSearchCV(estimator=pipeline,
        param_grid=parameters,
        cv=5)
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    return classification_report(Y_test,y_pred,target_names = category_names)


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
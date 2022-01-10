import json
import plotly
import pandas as pd
import numpy as np
from collections import Counter

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# save all possible POS tags
pos_tags = ['CC','CD','DT','EX','FW','IN','JJ','JJR',
 'JJS','LS','MD','NN','NNP','NNPS','NNS',
 'PDT','POS','PRP','PRP$','RB','RBR','RBS',
 'RP','SYM','TO','UH','VB','VBD','VBG','VBN',
 'VBP','VBZ','WDT','WP','WP$','WRB']
from nltk import TweetTokenizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine

app = Flask(__name__)

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

def tokenize(text):
    '''
    tokenizing function
    input: text 
    output: cleaned text
    '''
    # create tokens by tokenizing words
    tokens = word_tokenize(text)
    # initalize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # initializa list to store tokens
    clean_tokens = []
    for tok in tokens:
        # lemmatize, lower case, and remove white strips
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # append cleaned token to the list
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/{}'.format('DisasterResponse.db'))
df = pd.read_sql_table('disaster_tweets', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    ## extract data needed for visuals
    # drop non-label columbs
    # get counts of number of labels per tweet
    label_counts = df.drop(['message','original','genre', 'VADER'], axis=1).sum(axis = 1).value_counts()
    # stores number of labels per tweet
    label_names = list(label_counts.index)

    ## plot histogram of sentiment
    # make bins of size 0.05 from -1 (most sad) to 1 (most happy)
    bins = np.linspace(-1,1,21)
    # split sentiment into the bins and groupby each bin
    sentiment = df['VADER'].groupby(pd.cut(df['VADER'], bins=bins)).size()
    # convert index from object to string
    sentiment.index = sentiment.index.map(str)
    # store variables for plot
    sentiment_x = sentiment.index
    sentiment_y = sentiment.values
    
    # create visuals for home page
    graphs = [
        {
            # bar graph of number of labels per tweet
            'data': [
                Bar(
                    x=label_names,
                    y=label_counts
                )
            ],
            
            # add labels to graph
            'layout': {
                'title': 'Number of Labels per Tweet',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "# of Labels"
                }
            }
        },
        {   # histogram of sentiment
            'data': [
                Bar(
                    x=sentiment_x,
                    y=sentiment_y
                )
            ],
            
            # add labels to graph
            'layout': {
                'title': 'VADER Sentiment',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Sentiment"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
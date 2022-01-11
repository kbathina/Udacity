# Disaster Response Pipeline Project

The README file includes a summary of the project, code dependencies, and an explanation of the files in the repository, and instructions on how to run the Python scripts and web app.

### Description:

This project is a Udacity nanodegree project involving machine learning. Using labelled tweets, I build a machine learning model. This is displayed in a web page with interactive visualizations and an option to test out new tweets.

![Landing Page](disaster_homepage.png)

### Python Dependencies:

1. Machine Learning - Sciki-Learn, NLTK
2. Data Storage : SQLalchemy, Pickle 
3. Visualization and Webpage - Flask, Plotly

### File Structure and Descriptions:

1. `app` - directory containing files related to running the Flask web app
    1. `run.py` - python script to run ML model, produce visualizations, and run web app
    2. `templates.` - directory containing html files for the web page
        1. `go.html` -  extension functions for the main page
        2. `master.html` - main tml page displaying the web app
2. `data/` - directory containing raw data and processing script 
    1. `disaster_categories.csv` - dataset containing categories each tweet is labelled as
    2. `disaster_messages.csv` - dataset containing the original message, the translation, and the genre
    3. `process_data.csv` - script to load, clean, and save data into an SQLite database stored in the same directory
3. `models` - directory containing script to build and save ML model
    1. `train_classifier.py` - loads data, builds classifier, and saves ML model in the same directory


### How to Run:
1. Run the following commands in the project's root directory.

    - ETL Pipeline - loads, cleans, and saves data
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - ML Pipeline - build, run, and save model
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the `app` directory to run the web app
    ```python run.py```

3. Go to http://0.0.0.0:3001/

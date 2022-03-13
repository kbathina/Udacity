# Understanding Musical Features in Various Genres

![](https://miro.medium.com/max/1400/1*7RptC87l1-rSKZtBf5DTCg.jpeg)

Music is typically classified into genres, such as pop, rock, or rap but defining these genres is very difficult. For example, can rap songs also be classified as pop? What if a rock song has rap in it? 

Because of the difficulty of human classification, I look into using machine learning to identify patterns in genres using various measurements from songs.

## Data

I downloaded a music dataset from [Kaggle](https://www.kaggle.com/purumalgi/music-genre-classification). This data includes 17,997 songs that are categorized into 1 of 11 genres. Each song has the metadata provided as well as musical features. 

### Genres

In total, the dataset includes 11 features with the below frequencies:

| Genre      | Frequency |
| ----------- | ----------- |
| Rock      | 4949       |
| Indi Alt   | 2587        |
| Pop      | 2524       |
| Metal   | 1854        |
| HipHop      | 1447       |
| Alt_Music   | 1373        |
| Blues      | 1272       |
| Acoustic/Folk   | 625        |
| Instrumental      | 576       |
| Bollywood   | 402        |
| Country      | 387       |

There is a large class imbalance that will have to be accounted for when building the machine learning classifiers.

### Musical Features

There are 15 features in the dataset. A description from [Spotify](https://developer.spotify.com/discover/) of each feature is described below. 


| Feature      | Description | Feature      | Description |
| ----------- | ----------- | ----------- | ----------- |
| Artist Name      |  name of the artist| Track Name|name of the song|
| Popularity      | how popular the song is|Danceability| how well a song is suited for dancing|
| Energy      |a measure of intensity and activity in the song|Key|the musical key the song is in|
| Loudness      | the average decibels of the song|Speechiness|the presence of spoken word in the song|
| Acousticness      | how acoustic the song is|Instrumentalness|how much of the song consists of instrumentals|
| Liveness   | the probability of a song being recorded with a live audience| Valence |the amount of positivity in the song|
| Tempo |the speed of the song| Duration_in min/ms|the length of the song|
| Time_signature|the time signature of the song based on quarter notes | | |

In order to have consistent data types, I limit features to those that are continuous Basic statistics of the remaining features are shown below. 

|| Popularity      | Danceability | Energy      | Loudness | Speechiness      | Acousticeness | Instrumentalness      | Liveness | Valence      | Tempo | Duration_in min/ms      |
| ----------- | ----------- | ----------- | ----------- |----------- | ----------- | ----------- | ----------- |----------- | ----------- | ----------- | ----------- |
minimum	|1.000000	|0.059600	|0.000020	|-39.95200	|0.022500	|0.000000	|0.000001	|0.01190	|0.018300	|30.557000	|5.016500e-01
mean	|44.512124	|0.543433	|0.662777	|-7.91066	|0.079707	|0.247082	|0.177562	|0.19617	|0.486208	|122.623294	|2.007445e+05
median	|44.000000	|0.545000	|0.700000	|-7.01600	|0.047400	|0.081400	|0.003910	|0.12900	|0.481000	|120.065500	|2.091600e+05
maximum	|100.000000	|0.989000	|1.000000	|1.35500	|0.955000	|0.996000	|0.996000	|1.00000	|0.986000	|217.416000	|1.477187e+06

### Fixing the Time Feature

The distribution of many of the features make sense except for Duration_in min/ms.

<img src="images/time.png" width="400" height="300" />

This bimodel distribution explains the strange feature name; Some of the songs are in milliseconds and some are in minutes. To fix this, I split the distribution in half and convert all times to minutes. 

### Missing Data

3 Features contain missing features. 
* Popularity: 428
* Key: 2014
* Instrumentalness: 4377

Removing rows with missing data reduces the data size to 65% of the original size.

## Exploratory Data Analysis

When looking at the distribution of features, some interesting patterns emerge. Distributions are shown in the Appendix.

1. Expectedly, pop music has the most popular songs. What is surprising is that, on average, country music is more popular.
2. Hip Hop and Pop are the most danceable music. I would expect Bollywood music to be ranked higher; this could be due to its lack of western popularity.
3. Metal has the most energy and is the most loud genre.
4. Hip Hop is the most speech based genre while instrumental contains the most instrumentalness and is the most acoustic.
5. There is no significant pattern with tempo and genre.
6. Bollywood music tends to be the longest. I am surprised that instrumental music is not higher ranked; this could be due to an underrepresentation of classical music.

## Classification

My strategy to build a good classifier was to try many different algorithms and parameters. Because of the small amount of data and features, this is a reasonable approach.

### Models

I consider the below models and optimize for parameters using a grid search. Test and training splits were built accounting for the class imbalance.

1. K Nearest Neighbors (KNN)
2. Logistic Regression (LogReg)
3. Support Vector Machines (SVC)
4. Random Forest (RandForest)
5. ADA Boost (ADA)
6. Gaussian Naive Bayes (GNB)

### Results

I plotted the validation accuracy across 10 independent runs of each type of classifier.ADA and GNB had the lowest performance at around 37%. KNN and LogReg had about 43% accuracy while SVC and RandForest were the best performrs at 46%. There is about a 9% chance to randomly guess the correct genre.

<img src="images/results.png" width="500" height="300" />

I look at the F1 score to to see how well the genres were classified. Because of the many classes, the total score represents a weighted average. In all classifiers, the instrumental genre had the highest F1 score and thus the best precision and recall.

<img src="images/f1.png" width="500" height="300" />

## Appendix

<img src="images/Feature Distribution Across Genre.png"/>
# Understanding Musical Features in Various Genres

![](https://miro.medium.com/max/1400/1*7RptC87l1-rSKZtBf5DTCg.jpeg)

Music is typically classified into genres, such as pop, rock, or rap but defining these genres is very difficult. For example, can rap songs also be classified as pop? What if a rock song has rap in it? 

Because of the difficulty of human classification, I look into using machine learning to identify patterns in genres using various measurements from songs.

## Music Data

I downloaded a music dataset from (Kaggle)[https://www.kaggle.com/purumalgi/music-genre-classification]. This data includes songs that are categorized into 1 of 11 genres. Each song has the metadata provided as well as musical features. 

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

## Classification

### Results
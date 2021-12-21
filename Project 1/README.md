The following libraries were used in this analysis:

1. matplotlib
2. numpy
3. re
4. pandas
5. datetime
6. string
7. math
8. collections
9. sklearn
10. nltk

The files in the repository include three `.csv` data files in the `data/` directory as we all a jupyter notebook (`Boston AirBnB.ipynb`) that contains the analysis.

The motivation for doing this project is to study patterns in AirBnBs in Boston. My main results are summaized below:

1. There is a temporal pattern in when Boston AirBnBs are booked. Thursdays, Fridays, and Saturdays have more bookings than the other days of the week. A seasonal temporal pattern can be properly gleaned given more data.
2. Prices show a similar pattern to bookings with a slight increase for Thursdays, Fridays, and Saturdays.
3. Using TF-IDF scores from the neighorhood descriptions, I was able to isolate words that are important and describbe each neighborhood. I also show words that uniquely do not describe each neighborhood well.

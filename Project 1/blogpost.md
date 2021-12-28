# Finding patterns in AirBnB postings: A Boston Case Study

![](https://www.planetware.com/photos-large/USMA/boston-massachusetts-united-states-freedom-trail.jpg)

AirBnBs have been growing in popularity across the world ranging from a small room in an apartment to large beach side mansions. Renting an AirBnB is a simple process; log into the website, find a place, and then book!

Of course, we all know that finding a place is not that easy. Once you narrow down the location and time, you still need to find an AirBnB that has the right number of rooms, close to points of interest, and many other amenities. Besides looking at reviews, the actual post also plays a big role in the attractiveness of a unit. In this analysis, I explore some commonalities of AirBnBs, focused in Boston. 

# Boston AirBnB Data

The data for this analysis was downloaded from [Kaggle](https://www.kaggle.com/airbnb/boston). There are a total of three files. 

`calender.csv` contains around 1.3 million availabilies of 3585 listings from Septembber 6 2016 to September 5 2017. A flag indicates if the listing was booked for a given day. There is also a price for AirBnBs that were not booked.

`listings.csv` contains information for each of the 3585 listings such as host information, listing description, and review score.

`reviews.csv` contains arouund 68 thousand reviews for the same listings. This data includes the date of the review but **not** the booking date.

# What are the patterns in booking times?

![Overall Number of Bookings in Boston](images/bookings.png)

The above plot shows the number of bookings over time. The first thing to notice is that there is a large drop in bookings from October 2016 to Jan 2017. While it is tempting to say this is due to less bookings in the winter, this is impossible to confirrm without more longitudinal data. This is especially evident from March onwards since there is no increase in bookings during the summer months.

As a side note, the Boston Marathon shows how large of an effect a single event can have in the number of bookings. The day of the marathon had 12% more bookings than the previous. 

## Weekly Patterns

I typically book AirBnBs when I am on vacation (Friday and Saturday night) and this is probably true for a majority of people. In order to test this, I look at booking rates across each day of the week (Mon - Sun).

![Bookings by Day of Week](images/DOW-bookings.png)

The blue line indicates the average number of bookings across all 3585 listings. As expected, Friday and Saturday were the highest followed by Thursday. The standard workdays showed lower rates of booking. This is further highlighted by the individual listings shown in the red lines.

# What are the patterns in pricing?
# What words are used to describe the different neighbborhoods in Boston?

How I answered it

the answer
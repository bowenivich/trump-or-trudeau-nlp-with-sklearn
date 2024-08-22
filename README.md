# Trump or Trudeau? NLP with scikit-learn

## Introduction

There is a variety of powerful tools for Natural Language Processing (NLP) to analyze and interpret textual data. This project serves as a practical application to apply the simpliest NLP technique using an interest dataset. The primary goal is to distinguish the tweets by Donald Trump or Justin Trudeau with a simple yet effective approach. 

## Rundown

### Import Dataset

This is an interesting dataset. It is shared by Moez Ali, an amazing professor who is teaching Predictive Modelling and Big Data Analytics at Queen's University. It contains 400 tweets by Donald Trump and Justin Trudeau. 

```
import pandas as pd

df = pd.read_csv('tweets_trump_trudeau.csv')
df = df.set_index('id')
```

### A Quick Look

Tweets from Donald Trump. 
| id  | author          | status                                            |
|-----|-----------------|---------------------------------------------------|
| 157 | Donald J. Trump |                 #JFKFiles https://t.co/AnPBSJFh3J |
| 152 | Donald J. Trump | After strict consultation with General Kelly, ... |
| 105 | Donald J. Trump | The United States will be immediately implemen... |
| 114 | Donald J. Trump | ....for the Middle Class. The House and Senate... |
| 130 | Donald J. Trump | Thank you @LuisRiveraMarin! https://t.co/BK7sD... |

Tweets from Justin Trudeau. 
| id  | author         | status                                            |
|-----|----------------|---------------------------------------------------|
| 345 | Justin Trudeau | RT @PMcanadien: En direct: le PM Trudeau souli... |
| 276 | Justin Trudeau | Merci à Nguyen Cong Hiep, du consulat canadien... |
| 336 | Justin Trudeau | Today, I spoke with Governor @GregAbbott_TX to... |
| 302 | Justin Trudeau | This afternoon, I met with Vietnam’s Secretary... |
| 323 | Justin Trudeau | RT @PattyHajdu: Focusing on prevention, increa... |

### Split for Training and Validation

I used 20% of the dataset for validation to see how well the model is performing. 
```
from sklearn.model_selection import train_test_split

train_x, valid_x, train_y, valid_y = train_test_split(df['status'], df['author'], test_size=0.2, random_state=42)
```

### Training

Since this project is for fun and the dataset is not huge, I am just using `TfidfVectorizer` instead of more advanced techniques.
```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer()
train_x_vec = vectorizer.fit_transform(train_x)
valid_x_vec = vectorizer.transform(valid_x)

model = LogisticRegression()
model.fit(train_x_vec, train_y)

y_pred = model.predict(valid_x_vec)
```

## Results

This logistic regression model achieved an accuracy score of 0.8875. It is safe to say that the model is performing very well. 
```
from sklearn.metrics import accuracy_score

accuracy_score(valid_y, y_pred)
```
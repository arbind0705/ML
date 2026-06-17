import numpy as np 
import pandas as pd

credits_df = pd.read_csv("credits.csv")
movies_df = pd.read_csv("movies.csv")

credits_df
movies_df

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

credits_df.head()
movies_df.tail()

movies_df = movies_df.merge(credits_df, on='title')
movies_df.shape
movies_df.head()

movies_df.info()

movies_df = movies_df[['movie_id','title','overview', 'genres','keywords','cast','crew']]

movies_df.head()
movies_df.info()

movies_df.isnull().sum()
movies_df.dropna(inplace=True)

movies_df.duplicate().sum()
movies_df.duplicate()

import ast
def convert (obj):
    L = []
    for i in ast.literal_equal():
        L.append()

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import nltk
import re
from nltk.corpus import stopwords
import string
import openpyxl

df = pd.read_excel("D:\Code\Python\Datasheet CERDAS\data siap pakai.xlsx")
#data = data.info()
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y',dayfirst=True)
#print(df['TANGGAL'].head())
df['PERMASALAHAN_USER'] = df['PERMASALAHAN_USER'].str.lower()
df['PERMASALAHAN_USER'] = df['PERMASALAHAN_USER'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
print(df['PERMASALAHAN_USER'].head())

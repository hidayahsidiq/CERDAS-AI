import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import openpyxl


df = pd.read_excel("C:/Users/sidiq/Belajar Python/CERDAS AI/data siap pakai.xlsx")
#print(df.shape)
#print(df.head())

complaints_df = df.filter(["KETERANGAN_BERSIH","KATEGORI"],axis = 1)
print(complaints_df.head())

print(complaints_df.isnull().sum())

complaints_df.dropna(inplace=True)
print(complaints_df.shape)
print(complaints_df.isnull().sum())

complaints_df = complaints_df[:1000]
print(complaints_df.KATEGORI.value_counts().plot(kind='pie',autopct='%1.0f%%',figsize=(12,8)))

def clean_text(text):

    complaints = []

    for comp in text:
        # remove special characters
        comp = re.sub(r'\W', ' ', str(comp))

        # remove single characters
        comp  = re.sub(r'\s+[a-zA-Z]\s+', ' ', comp )

        # Remove single characters from the beginning
        comp  = re.sub(r'\^[a-zA-Z]\s+', ' ', comp)

        # Converting to Lowercase
        comp  = comp.lower()

        complaints.append(comp)

    return complaints

complaints = clean_text(list(complaints_df['KETERANGAN_BERSIH']))
#print(complaints)

tfid_conv = TfidfVectorizer(max_features=3000, min_df=10, max_df=0.7,stop_words=stopwords.words('indonesian'))
X = tfid_conv.fit_transform(complaints).toarray()

complaints_df['KATEGORI'] = complaints_df['KATEGORI'].astype('category')
y = list(complaints_df['KATEGORI'].cat.codes)
print(y)
"""
nltk.download('stopwords')
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y',dayfirst=True)
print(df['TANGGAL'].head())
df['KETERANGAN_BERSIH'] = df['KETERANGAN_BERSIH'].astype(str)
df['KETERANGAN_BERSIH'] = df['KETERANGAN_BERSIH'].str.lower()
df['KETERANGAN_BERSIH'] = df['KETERANGAN_BERSIH'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
print(df['KETERANGAN_BERSIH'].head())
"""

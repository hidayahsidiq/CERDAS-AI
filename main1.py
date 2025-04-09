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
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_excel("C:/Users/sidiq/Belajar Python/CERDAS AI/data siap pakai.xlsx")
#print(df.shape)
#print(df.head())

complaints_df = df.filter(["KETERANGAN_BERSIH","HASIL PREDIKSI"],axis = 1)
print(complaints_df.head())

print(complaints_df.isnull().sum())

complaints_df.dropna(inplace=True)
print(complaints_df.shape)
print(complaints_df.isnull().sum())

print(complaints_df['HASIL PREDIKSI'].value_counts()).plot(kind='pie',autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of Complaint Types')
"""
nltk.download('stopwords')
df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], format='%d-%m-%Y',dayfirst=True)
print(df['TANGGAL'].head())
df['KETERANGAN_BERSIH'] = df['KETERANGAN_BERSIH'].astype(str)
df['KETERANGAN_BERSIH'] = df['KETERANGAN_BERSIH'].str.lower()
df['KETERANGAN_BERSIH'] = df['KETERANGAN_BERSIH'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
print(df['KETERANGAN_BERSIH'].head())
"""

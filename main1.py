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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import string
import openpyxl

df = pd.read_excel("D:\Code\Python\Datasheet CERDAS\data siap pakai.xlsx")
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
category_mapping = dict(enumerate(complaints_df['KATEGORI'].cat.categories))
print("Category Mapping : ",category_mapping)
y = list(complaints_df['KATEGORI'].cat.codes)
print(y)

X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.012, random_state=42)
classifier = RandomForestClassifier(n_estimators=500, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


print("classification_report",classification_report(y_test, y_pred))
print("akurasi skore",accuracy_score(y_test, y_pred))

user = input("Masukkan keluhan anda: ")
data = [user]
data = clean_text(data)
data = tfid_conv.transform(data).toarray()
pred = classifier.predict(data)

predict_category = category_mapping[pred[0]]
print("Prediksi kategori keluhan anda adalah: ", predict_category)

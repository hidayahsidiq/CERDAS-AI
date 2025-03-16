#this is based on aman kharwal page 
#pip install pandas
import pandas as pd
#pip install numpy
import numpy as np
#pip install -U scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
#pip install nltk
import nltk
import re
from nltk.corpus import stopwords
import string

data = pd.read_csv("G:\complaints\consumercomplaints.csv")
#print(data.head())

#drop unnamed column
data = data.drop("Unnamed: 0", axis=1)

#print(data.isnull().sum())
#drop NA value
data = data.dropna()

#print(data["Product"].value_counts())

nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words("english"))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["Consumer complaint narrative"] = data["Consumer complaint narrative"].apply(clean)

#split data into training dataset
data = data[["Consumer complaint narrative", "Product"]]
x = np.array(data["Consumer complaint narrative"])
y = np.array(data["Product"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.33, 
                                                    random_state=42)

#train data into stochastic Gradient descent Classification Algorithm
sgdmodel = SGDClassifier()
sgdmodel.fit(X_train,y_train)


#output to test aresult of training data
user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = sgdmodel.predict(data)
print(output)


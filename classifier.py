import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from joblib import dump

dataset = pd.read_csv('main.csv',encoding='utf-8')
print("########## File read ##########\n")
tweets = dataset["tweet_text"].values
classes = dataset["sentiment"].values
X_train, X_test, y_train, y_test = train_test_split(tweets, classes, test_size=0.3, random_state=42)
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(X_train)
print("########## Dictionary built ##########\n")
test_data_features = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(train_data_features, y_train)
MNBpredict = model.predict(test_data_features)
print("Accuracy: {0:.2f}%".format(metrics.accuracy_score(y_test, MNBpredict)*100))
dump(model, 'model.joblib')

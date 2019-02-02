from flask import Flask, request, json, jsonify
from joblib import load
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

@app.route("/classify", methods=['POST'])
def classify():
    dataset = pd.read_csv('main.csv',encoding='utf-8')
    print("########## File read ##########\n")
    tweets = dataset["tweet_text"].values
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,
                                 max_features = 5000)
    vectorizer.fit_transform(tweets)

    model = load('model.joblib')
    test = [request.form.get('phrase')]

    X = vectorizer.transform(test)
    MNBpredict = model.predict(X)
    print(MNBpredict)
    data = jsonify(sentiment = MNBpredict[0])
    return data

if __name__ == "__main__":
   app.run(port=5000, debug=True)

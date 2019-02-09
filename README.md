# sentimental-analysis-server

This is a simple and powerfull Machine Learning sentiment analysis service. You can find a simple tutorial of how it was built [here](https://medium.com/@vini.bandeira.vb/creating-your-own-sentiment-analysis-service-to-classify-portuguese-phrases-eb2fb6613eb1)

## To run

Clone the repository:
```
git clone https://github.com/vinicius-b/sentimental-analysis-server.git
```
And install requirements:
```
pip3 install -r requirements.txt
```
To run the service using default model, you just have to type:
```
python3 flask_app.py
```
### Sending a phrase to classify
```
curl -X POST -F 'phrase=YOUR_PHRASE' http://localhost:5000/classify
```

## Building a new model

To build a new model you need to download dataset [here](https://drive.google.com/file/d/11pKJEeJ44qL1cVmllZyd4o26HkYXG3Yn/view?usp=sharing), unzip it into the application path, delete `model.joblib` and run `classifier.py`:
```
python3 classifier.py
```

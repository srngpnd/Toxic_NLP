import pandas as pd
from flask import Flask, request, jsonify, render_template
from model import simple_tokenizer
import pickle
import json

app = Flask(__name__)
toxic_model = pickle.load(open('toxic_model.pkl', 'rb'))
severe_toxic_model = pickle.load(open('severe_toxic_model.pkl', 'rb'))
obscene_model = pickle.load(open('obscene_model.pkl', 'rb'))
threat_model = pickle.load(open('threat_model.pkl', 'rb'))
insult_model = pickle.load(open('insult_model.pkl', 'rb'))
identity_hate_model = pickle.load(open('identity_hate_model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route('/predict',methods=['POST'])
def predict():
    comment = request.args["comment"]
    a = tfidf_vectorizer.transform([comment])

    toxic_prediction = toxic_model.predict_proba(a).tolist()
    severe_toxic_prediction = severe_toxic_model.predict_proba(a).tolist()
    obscene_prediction = obscene_model.predict_proba(a).tolist()
    threat_prediction = threat_model.predict_proba(a).tolist()
    insult_prediction = insult_model.predict_proba(a).tolist()
    identity_hate_prediction = identity_hate_model.predict_proba(a).tolist()

    output = {'toxic': toxic_prediction[0][1],
              'severe_toxic': severe_toxic_prediction[0][1],
              'obscene': obscene_prediction[0][1],
              'threat': threat_prediction[0][1],
              'insult': insult_prediction[0][1],
              'identity_hate': identity_hate_prediction[0][1]}

    return json.dumps( output )

if __name__ == "__main__":
    app.run(debug=True)
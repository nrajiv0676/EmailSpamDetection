import numpy as np
import pandas as pd
import pickle

from flask import Flask, render_template,request
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)

@app.route("/")
def index():
    return  render_template("index.html")


@app.route("/result", methods=['POST','GET'])
def result():
    print('Loading the model...')
    model = pickle.load(open('spamclassify.pkl', 'rb'))
    count_vect = pickle.load(open('spamclassify_cv', 'rb'))
    print('Model is loaded')

    email_text = str(request.form['emailtext'])
    print('email_text: ' + email_text)

    corpus = []
    corpus.append(email_text)
    print('corpus:: ')
    print(corpus)

    print('start prediction ...')
    test_result = model.predict(count_vect.transform(corpus))
    print('test result :: ')
    print(test_result)

    if test_result==0:
        return render_template('noSpam.html')
    else:
        return render_template('spam.html')

if __name__ == '__main__':
    app.run(debug=True)
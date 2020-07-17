from flask import render_template
# from app import app
from flask import request

from flask_cors import CORS
from flask import Flask

import bert_load
import get_sentiment
import bert_viz
import rating_prediction

app = Flask(__name__)
CORS(app)

@app.route('/')
@app.route('/classify', methods=['GET'])
def index():
    return render_template('index.html', weights=[], words=[], prediction="", rating_prediction = [], rating_prediction_probability=[], score = "", display=False)

@app.route('/classify', methods=['POST'])
def classification():
    reviewText = request.form['reviewText']
    input_ids , attention_mask= bert_load.method_load(reviewText)
    sentiment = get_sentiment.run(input_ids, attention_mask)
    weights, words = bert_viz.result(reviewText)
    rating_prediction_probability, rating_prediction_value = rating_prediction.run_rating(input_ids ,attention_mask)
    rating_value = 1
    score = 0
    for value in rating_prediction_probability:
        score += rating_value*value
        rating_value += 1
    score = round((score/100),2)

    return render_template('index.html', weights=weights, words=words, prediction=sentiment, rating_prediction=rating_prediction_value, rating_prediction_probability = rating_prediction_probability, score = score, display=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug='True')


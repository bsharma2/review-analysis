from flask import request
from flask import render_template
from flask import flash
from flask import jsonify
import os
# from app import app
from flask_cors import CORS
from flask import Flask

app = Flask(__name__)
CORS(app)

@app.route('/')
@app.route('/classify', methods=['GET'])
def index():
    return render_template('index.html', weights=[], words=[], prediction="", rating_prediction = [], rating_prediction_probability=[], score = "", display=False)

@app.route('/classify', methods=['POST'])
def classification():
    reviewText = request.form['reviewText']
    import get_sentiment
    sentiment = get_sentiment.run(reviewText)
    import bert_viz
    weights, words = bert_viz.result(reviewText)
    import rating_prediction
    rating_prediction_probability, rating_prediction = rating_prediction.run_rating(reviewText)
    # import analysis_model
    # sentiment = analysis_model.run(reviewText)
    # weights, words = analysis_model.result(reviewText)
    # rating_prediction_probability, rating_prediction = analysis_model.run_rating(reviewText)

    rating_value = 1
    score = 0
    for value in rating_prediction_probability:
        score += rating_value*value
        rating_value += 1
    score = round((score/100),2)

    return render_template('index.html', weights=weights, words=words, prediction=sentiment, rating_prediction=rating_prediction, rating_prediction_probability = rating_prediction_probability, score = score, display=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug='True')
    # app.run(host='0.0.0.0', port=5000, debug='True')


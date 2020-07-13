from flask import render_template
from app import app
#import bert_viz
#import sentiment_analysis
from flask import request


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
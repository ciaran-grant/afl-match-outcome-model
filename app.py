from afl_match_outcome_model.predict.predict import predict_outcome
from flask import Flask, request

app = Flask(__name__)

@app.route("/model/outcome/predict", methods=["POST"])
def predict():
    match_id = request.json['Match_ID']
    probas = predict_outcome(match_id).tolist()
    return {'prediction':probas}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
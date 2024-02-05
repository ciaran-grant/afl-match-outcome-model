import os
print(os.environ)

import joblib
from afl_match_outcome_model.predict.predict import predict_outcome
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/model/outcome/predict", methods=["POST"])
def predict():
    
    model_file_path = "model_outputs/match_outcome_xgb.joblib"
    super_xgb = joblib.load(model_file_path)
    
    match_id_list = request.json['Match_ID']
    result = {}
    for match_id in match_id_list:
        probas = predict_outcome(match_id, super_xgb)
        result[match_id] = probas
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    

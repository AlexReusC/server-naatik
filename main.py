from flask import Flask, request
from joblib import load
import pandas as pd

app = Flask(__name__)
prediction_model = load("dummy-model.joblib")

@app.route('/', methods=["POST"])
def run_models():
	if request.method == "POST" and request.files:
		df = pd.read_csv(request.files["data"])
		prediction = prediction_model.predict(df)
		return prediction.tolist()

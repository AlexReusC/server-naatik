import os
import uuid
from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
from joblib import load
import pandas as pd
import json

def get_probability_churn(probabilities):
	return list(map(lambda x: x[1], probabilities))

app = Flask(__name__)
CORS(app)
prediction_model = load("classification-model.joblib")

@app.route('/', methods=["POST"])
def run_models():
	if request.method == "POST" and request.files:
		#get parameters
		df = pd.read_csv(request.files["data"])
		slides = json.loads(request.form["slides"])
		threshold1, threshold2, threshold3 = slides["first-slide"], slides["second-slide"], slides["third-slide"]
		threshold1, threshold2, threshold3 = float(threshold1), float(threshold2), float(threshold3)

		#prediction
		prediction = prediction_model.predict_proba(df)
		prediction = get_probability_churn(prediction)

		#create file
		prediction_dataframe = pd.DataFrame({"Target": prediction})
		df = pd.concat([df, prediction_dataframe], axis=1)
		ui = str(uuid.uuid4())
		os.makedirs("breakdownPredictions", exist_ok=True)
		df.to_csv(f"breakdownPredictions/{ui}.csv")

		#aggregate little groups
		group1 = df[df["Target"] < threshold1]
		group2 = df[(df["Target"] >= threshold1) & (df["Target"] < threshold2)]
		group3 = df[(df["Target"] >= threshold2) & (df["Target"] < threshold3)]
		group4 = df[df["Target"] >= threshold3]
		group1_acc, group2_acc, group3_acc, group4_acc = group1.agg({"BILL_AMOUNT": "sum"}), group2.agg({"BILL_AMOUNT": "sum"}), group3.agg({"BILL_AMOUNT": "sum"}), group4.agg({"BILL_AMOUNT": "sum"})

		return jsonify({"ui": ui, "acc": {"group1": group1_acc.values[0], "group2": group2_acc.values[0], "group3": group3_acc.values[0], "group4": group4_acc.values[0]}}), 200

@app.route('/retrievecsv', methods=["GET"])
def retrieve_csv():
	if request.method == "GET" and request.args.get("ui"):
		csv_dir = "./breakdownPredictions"
		filename = request.args.get("ui")
		csv_file = f"{filename}.csv"
		return send_from_directory(csv_dir, csv_file), 200


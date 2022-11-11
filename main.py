import os
import uuid
from flask import Flask, request, send_from_directory, jsonify, url_for
from flask_cors import CORS
from joblib import load
import pandas as pd
import numpy as np
import json
from differences_images import create_images
from calculate_differences import get_differences
import plotly.express as px

# import ETL modules


from data_transformation.functions.transform_prediction import transform_df_predict


def get_probability_churn(probabilities):
	return list(map(lambda x: x[1], probabilities))

def make_etl_transformation(df):
	transform_df_predict(df)

def get_original_file_rows(df):
	column_names = list(df.columns.values)
	return column_names

app = Flask(__name__)
CORS(app)
prediction_model = load("./data_transformation/joblibs/telecom_churn_me/model/classification-model.joblib")
os.makedirs("static", exist_ok=True)
os.makedirs("static/images_differences", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

@app.route('/', methods=["POST"])
def run_models():


	if request.method == "POST" and request.files:
		#get parameters
		df = pd.read_csv(request.files["data"])
		#print(df)

		# make etl that create a file called transformed_new.csv
		make_etl_transformation(df)

		print("etl completed.")

		# read that transformed csv
		df_encoded = pd.read_csv('transformed_new.csv')
		print(df_encoded)

		slides = json.loads(request.form["slides"])
		threshold1, threshold2, threshold3 = slides["first-slide"], slides["second-slide"], slides["third-slide"]
		threshold1, threshold2, threshold3 = float(threshold1), float(threshold2), float(threshold3)
		threshold1, threshold2, threshold3 = threshold1 / 100, threshold2 / 100, threshold3 / 100


		#prediction
		prediction = prediction_model.predict_proba(df_encoded)
		prediction = get_probability_churn(prediction)

		#create file
		prediction_dataframe = pd.DataFrame({"Probabilidad de churn": prediction})

		df = df.join(prediction_dataframe)
		ui = str(uuid.uuid4())
		os.makedirs("breakdownPredictions", exist_ok=True)
		df.to_csv(f"breakdownPredictions/{ui}.csv")

		#aggregate little groups
		group1 = df[df["Probabilidad de churn"] < threshold1]
		group2 = df[(df["Probabilidad de churn"] >= threshold1) & (df["Probabilidad de churn"] < threshold2)]
		group3 = df[(df["Probabilidad de churn"] >= threshold2) & (df["Probabilidad de churn"] < threshold3)]
		group4 = df[df["Probabilidad de churn"] >= threshold3]
		group1_acc, group2_acc, group3_acc, group4_acc = group1.agg({"BILL_AMOUNT": "sum"}), group2.agg({"BILL_AMOUNT": "sum"}), group3.agg({"BILL_AMOUNT": "sum"}), group4.agg({"BILL_AMOUNT": "sum"})

		little_groups_counts = pd.DataFrame(data={"count": [len(group1), len(group2),len(group3.index), len(group4.index)]}, index=["sin churn", "churn bajo", "churn medio", "churn alto"])

		histogram_little_groups = px.bar(x=little_groups_counts.index, y=little_groups_counts["count"])
		pie_little_groups = px.pie(little_groups_counts, values=little_groups_counts["count"], names=little_groups_counts.index, title="Grupos")

		os.makedirs(f"static/graphs/{ui}", exist_ok=True)
		histogram_little_groups.write_image(f"static/graphs/{ui}/histogram.png")
		pie_little_groups.write_image(f"static/graphs/{ui}/pie.png")

		#Create images churn vs no churn
		churn = group1
		nochurn = df[df["Probabilidad de churn"] >= threshold1]

		differences = None
		state = "both"
		if len(churn.index) == 0:
			state = "nochurn"
		elif len(nochurn.index) == 0:
			state = "churn"
		else:
			create_images(churn, nochurn, ui)
			differences = get_differences(churn, nochurn)

		fileRows = get_original_file_rows(df)

		return jsonify({"ui": ui,"fileRows": fileRows, "acc": {"group1": round(group1_acc.values[0],2), "group2": round(group2_acc.values[0],2), "group3": round(group3_acc.values[0],2), "group4": round(group4_acc.values[0],2)}, "differences": differences, "state": state}), 200

@app.route('/retrievecsv', methods=["GET"])
def retrieve_csv():
	if request.method == "GET" and request.args.get("ui"):
		csv_dir = "./breakdownPredictions"
		filename = request.args.get("ui")
		csv_file = f"{filename}.csv"
		return send_from_directory(csv_dir, csv_file), 200

@app.route('/getdifferences', methods=["GET"])
def getdifferences():
	if request.method == "GET" and request.args.get("ui"):

		ui = request.args.get("ui")
		image_files = os.listdir(f'static/images_differences/{ui}')
		arr = []
		# loop over the image paths
		for image_file in image_files:
			url = url_for('static', filename=f'images_differences/{ui}/{image_file}')
			arr.append(url)
		return arr, 200

@app.route('/getgraphs', methods=["GET"])
def getgraphs():
	if request.method == "GET" and request.args.get("ui"):

		ui = request.args.get("ui")
		image_files = os.listdir(f'static/graphs/{ui}')
		arr = []
		# loop over the image paths
		for image_file in image_files:
			url = url_for('static', filename=f'graphs/{ui}/{image_file}')
			arr.append(url)
		return arr, 200
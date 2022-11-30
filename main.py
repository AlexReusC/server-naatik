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

from model_mlp_train import train_mlp

from data_transformation.functions.transform_model import transform_df_model
from data_transformation.functions.storage_configuration import configure_storage

def get_probability_churn(probabilities):
	return list(map(lambda x: x[1], probabilities))

def make_etl_transformation(original_name_dataset):
	transform_df_predict(original_name_dataset)

def train_model(target,filename, smote):
	train_mlp(target,filename, smote)

def get_thresholds(slides):
		threshold1, threshold2, threshold3 = slides["first-slide"], slides["second-slide"], slides["third-slide"]
		threshold1, threshold2, threshold3 = float(threshold1), float(threshold2), float(threshold3)
		threshold1, threshold2, threshold3 = threshold1 / 100, threshold2 / 100, threshold3 / 100
		return threshold1, threshold2, threshold3

def split_by_little_groups(df, threshold1, threshold2, threshold3):
		group1 = df[df["Probabilidad de churn"] < threshold1]
		group2 = df[(df["Probabilidad de churn"] >= threshold1) & (df["Probabilidad de churn"] < threshold2)]
		group3 = df[(df["Probabilidad de churn"] >= threshold2) & (df["Probabilidad de churn"] < threshold3)]
		group4 = df[df["Probabilidad de churn"] >= threshold3]
		return group1, group2, group3, group4

def add_bill_amount(group1, group2, group3, group4):
		group1['BILL_AMOUNT/PROBABILITIES'] = group1['Probabilidad de churn'] *  group1['BILL_AMOUNT']
		group2['BILL_AMOUNT/PROBABILITIES'] = group2['Probabilidad de churn'] *  group2['BILL_AMOUNT']
		group3['BILL_AMOUNT/PROBABILITIES'] = group3['Probabilidad de churn'] *  group3['BILL_AMOUNT']
		group4['BILL_AMOUNT/PROBABILITIES'] = group4['Probabilidad de churn'] *  group4['BILL_AMOUNT']
		return group1, group2, group3, group4

def get_little_groups_accs(group1, group2, group3, group4):
		group1_acc, group2_acc, group3_acc, group4_acc = group1.agg({"BILL_AMOUNT/PROBABILITIES": "sum"}), group2.agg({"BILL_AMOUNT/PROBABILITIES": "sum"}), group3.agg({"BILL_AMOUNT/PROBABILITIES": "sum"}), group4.agg({"BILL_AMOUNT/PROBABILITIES": "sum"})
		return group1_acc, group2_acc, group3_acc, group4_acc

def save_graphs_images(group1, group2, group3, group4, ui, i):
		little_groups_counts = pd.DataFrame(data={"count": [len(group1), len(group2),len(group3.index), len(group4.index)]}, index=["sin churn", "churn bajo", "churn medio", "churn alto"])

		histogram_little_groups = px.bar(x=little_groups_counts.index, y=little_groups_counts["count"])
		pie_little_groups = px.pie(little_groups_counts, values=little_groups_counts["count"], names=little_groups_counts.index, title="Grupos")

		os.makedirs(f"static/graphs/{ui}", exist_ok=True)
		os.makedirs(f"static/graphs/{ui}/{i}", exist_ok=True)
		pie_little_groups.write_image(f"static/graphs/{ui}/{i}/pie.png")
		histogram_little_groups.write_image(f"static/graphs/{ui}/{i}/histogram.png")

def differences_churn_nochurn(df, threshold1, ui, i):
		churn = df[df["Probabilidad de churn"] < threshold1].drop(["big_group", "Unnamed: 0", "Etiquetas de churn", "Probabilidad de churn"], axis = 1, errors="ignore")
		nochurn = df[df["Probabilidad de churn"] >= threshold1].drop(["big_group", "Unnamed: 0", "Etiquetas de churn", "Probabilidad de churn"], axis = 1, errors="ignore")

		differences = None
		state = "both"
		if len(churn.index) == 0:
			state = "nochurn"
		elif len(nochurn.index) == 0:
			state = "churn"
		else:
			create_images(churn, nochurn, ui, i)
			differences = get_differences(churn, nochurn)

		return state, differences

def get_original_file_rows(df):
	df = df.drop(["big_group", "Unnamed: 0", "Etiquetas de churn", "Probabilidad de churn"], axis = 1, errors="ignore")
	column_names = list(df.columns.values)
	return column_names

def add_probability_labels(prediction, thr1, thr2, thr3):
	prediction_dataframe = pd.DataFrame({"Probabilidad de churn": prediction})
	conditions = [
		(prediction_dataframe["Probabilidad de churn"] < thr1), 
		((prediction_dataframe["Probabilidad de churn"] >= thr1) & (prediction_dataframe["Probabilidad de churn"] < thr2)),
		((prediction_dataframe["Probabilidad de churn"] >= thr2) & (prediction_dataframe["Probabilidad de churn"] < thr3)),
		(prediction_dataframe["Probabilidad de churn"] >= thr3)
		]
	
	values = ["sin churn", "churn bajo", "churn medio", "churn alto"]

	prediction_dataframe["Etiquetas de churn"] = np.select(conditions, values)

	return prediction_dataframe 

def split_by_big_groups(df):
	big_groups = []
	unique_groups = df["big_group"].unique()
	unique_groups.sort()
	for group_id in unique_groups:
		tmp_group = df[df["big_group"] == group_id]
		big_groups.append(tmp_group)
	return big_groups

def mock_big_groups(df):
	np.random.seed(42)
	df["big_group"] = np.random.randint(1, 4, size=len(df))
	return df

def transformModel(original_name_dataset, target_column_name):
	transform_df_model(original_name_dataset, target_column_name)

app = Flask(__name__)
CORS(app)
os.makedirs("static", exist_ok=True)
os.makedirs("static/images_differences", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)
os.makedirs("raw_data/", exist_ok=True)

@app.route('/', methods=["POST"])
def run_models():


	if request.method == "POST" and request.files:
		#get parameters
		os.makedirs(f'./data_transformation/joblibs',exist_ok=True)
		print("name: ",request.files["data"].filename.split('.')[0])
		filename = request.files["data"].filename.split('.')[0]
		df = pd.read_csv(request.files["data"])
		df.to_csv(f'./raw_data/{filename}.csv')
		# make 1st step to training 

		target = json.loads(request.form["target"])
		slides = json.loads(request.form["slides"])
		
		configure_storage(f'./raw_data/{filename}.csv', filename)
		transformModel(filename,target)

		print("paso transform model")
		# make etl that create a file called transformed_new.csv
		#df = df.drop(['Unnamed: 0'])



		# file_name = json.loads(request.form["file_name"])
		threshold1, threshold2, threshold3 = get_thresholds(slides)

		#train model
		# agregar validacion de si existe la columna.
		train_model(target,filename, False)
		print("ya se entreno.")

		# cargamos modelo
		prediction_model = load("./data_transformation/joblibs/"+filename+"/model/mlp/mlp_model.joblib")

		# etl
		
		new_df = pd.read_csv('./data/'+filename+'/'+filename+'.csv')
		print(new_df.columns)
		new_df = new_df.drop([target], axis=1 )
		new_df.to_csv(f'./data/{filename}/{filename}_new.csv', index=False)
		make_etl_transformation(filename)

		# read that transformed csv
		df_encoded = pd.read_csv('./data/'+filename+'/'+filename+'_new_transformed.csv')

		print("paso etl")

		#prediction
		prediction = prediction_model.predict_proba(df_encoded)
		prediction = get_probability_churn(prediction)

		#create file
		prediction_dataframe = add_probability_labels(prediction, threshold1, threshold2, threshold3)
		df = df.join(prediction_dataframe)

		df = mock_big_groups(df)

		ui = str(uuid.uuid4())
		os.makedirs("breakdownPredictions", exist_ok=True)
		df.to_csv(f"breakdownPredictions/{ui}.csv")

		big_groups = split_by_big_groups(df)


		info = []
		for i, group in enumerate(big_groups):
			print("big_group", group)
			little_group1, little_group2, little_group3, little_group4 = split_by_little_groups(group, threshold1, threshold2, threshold3)
			little_group1, little_group2, little_group3, little_group4 = add_bill_amount(little_group1, little_group2, little_group3, little_group4)
			little_group1_acc, little_group2_acc, little_group3_acc, little_group4_acc = get_little_groups_accs(little_group1, little_group2, little_group3, little_group4)
			save_graphs_images(little_group1, little_group2, little_group3, little_group4,ui, i+1)
			state, differences = differences_churn_nochurn(group, threshold1, ui, i+1)
			info.append(
				{
					"i": i+1,
					"acc": 
					{ 
						"group1": round(little_group1_acc.values[0],2),
						"group2": round(little_group2_acc.values[0],2),
						"group3": round(little_group3_acc.values[0],2),
						"group4": round(little_group4_acc.values[0],2)
					},
					"differences": differences,
					"state": state,
					"total":
					{
						"group1": len(little_group1),
						"group2": len(little_group2),
						"group3": len(little_group3),
						"group4": len(little_group4),
					},
					"all_groups": len(group)

				} )

		fileRows = get_original_file_rows(df)

		return jsonify({"ui": ui, "fileRows": fileRows, "info": info}), 200

@app.route('/retrievecsv', methods=["GET"])
def retrieve_csv():
	if request.method == "GET" and request.args.get("ui"):
		csv_dir = "./breakdownPredictions"
		filename = request.args.get("ui")
		csv_file = f"{filename}.csv"
		return send_from_directory(csv_dir, csv_file), 200

@app.route('/getdifferences', methods=["GET"])
def getdifferences():
	if request.method == "GET" and request.args.get("ui") and request.args.get("i"):

		ui = request.args.get("ui")
		i = request.args.get("i")
		image_files = os.listdir(f'static/images_differences/{ui}/{i}')
		data = dict()
		# loop over the image paths
		for image_file in image_files:
			url = url_for('static', filename=f'images_differences/{ui}/{i}/{image_file}')
			filename = os.path.splitext(os.path.basename(url))[0]
			data[filename] = url
		return data, 200

@app.route('/getgraphs', methods=["GET"])
def getgraphs():
	if request.method == "GET" and request.args.get("ui") and request.args.get("i"):

		ui = request.args.get("ui")
		i = request.args.get("i")
		image_files = os.listdir(f'static/graphs/{ui}/{i}')
		arr = []
		# loop over the image paths
		for image_file in image_files:
			url = url_for('static', filename=f'graphs/{ui}/{i}/{image_file}')
			arr.append(url)
		return arr, 200
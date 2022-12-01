import pandas as pd

def categorical_difference(churn, nochurn, variable):
	churn_count_by_class = churn.groupby(variable)[variable].count()
	nochurn_count_by_class = nochurn.groupby(variable)[variable].count()

	differences = pd.DataFrame({"churn": churn_count_by_class, "no churn": nochurn_count_by_class}).reset_index().fillna(0)
	max_row, max_val = None, 0
	for index, row in differences.iterrows():
		if row["churn"] == 0 or row["no churn"] == 0:
			row["churn"] = row["churn"] + 1
			row["no churn"] = row["no churn"] + 1

		difference = row["churn"] / row["no churn"]
		if difference > max_val:
			max_row, max_val = row, difference 

	if max_row is None:
		return f"Sin diferencias en {variable}"
	else:
		difference = round(difference, 2)
		return f"El grupo con churn tiene {difference} veces de diferencia con respecto al grupo de no churn de {max_row[variable]} en {variable}"

def noncategorical_difference(churn, nochurn, variable):
	difference = churn[variable].mean() / nochurn[variable].mean()
	text = f"El grupo con churn tiene en promedio {round(difference, 2)} veces de diferencia con respecto al grupo con no churn en {variable}"
	return text

def get_differences(churn, nochurn):
	data = dict()
	for col_name, col_type in zip(churn.columns, churn.dtypes):
		if col_type == "object":
			data[col_name] = categorical_difference(churn, nochurn, col_name)
		elif col_type == "float64" or col_type == "int64":
			data[col_name] = noncategorical_difference(churn, nochurn, col_name)
	return data

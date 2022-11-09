import pandas as pd

def categorical_difference(churn, nochurn, variable):
	churn_count_by_class = churn.groupby(variable)[variable].count()
	nochurn_count_by_class = nochurn.groupby(variable)[variable].count()

	differences = pd.DataFrame({"churn": churn_count_by_class, "no churn": nochurn_count_by_class}).reset_index().fillna(0)
	max_row, max_val = "", 0
	for index, row in differences.iterrows():
		difference = row["churn"] - row["no churn"]
		avg = (row["churn"] + row["no churn"]) / 2
		percentage_difference = (difference / avg) * 100

		if abs(percentage_difference) > abs(max_val):
			max_row, max_val = row, percentage_difference

	text = f"El grupo con churn tiene {abs(percentage_difference)} {'mas' if max_val > 0 else 'menos'} de {max_row[variable]} en {variable} que el grupo sin churn"
	return text


def noncategorical_difference(churn, nochurn, variable):
	difference = churn[variable].mean() - nochurn[variable].mean()
	avg = (churn[variable].mean() + nochurn[variable].mean()) / 2

	percentage_difference = (difference / avg) * 100
	text = f"El grupo con churn tiene {abs(percentage_difference)} {'mas' if percentage_difference > 0 else 'menos'} de {variable} que el grupo sin churn"
	return text

def get_differences(churn, nochurn):
	bill_amount_text = noncategorical_difference(churn, nochurn, "BILL_AMOUNT")
	years_stayed_text = noncategorical_difference(churn, nochurn, "Years_stayed")

	party_gender_text = categorical_difference(churn, nochurn, "PARTY_GENDER_CD")

	return {"BILL_AMOUNT": bill_amount_text, "Years_stayed": years_stayed_text, "PARTY_GENDER_CD": party_gender_text}

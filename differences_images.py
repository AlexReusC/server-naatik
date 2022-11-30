
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_dataframe_for_plot_nocategoric(variable, churn, nochurn):
    bill_amount_churn = churn[variable].describe()[["min", "25%", "50%", "75%", "max"]]
    bill_amount_no_churn = nochurn[variable].describe()[["min", "25%", "50%", "75%", "max"]]

    difference_bill = pd.DataFrame({"churn": bill_amount_churn, "no churn": bill_amount_no_churn}).reset_index()
    difference_bill.rename(columns={'index':variable}, inplace=True)
    return difference_bill

def get_dataframe_for_plot_categoric(variable, churn, nochurn):
    party_gender_cd_churn_count = churn.groupby(variable)[variable].count()
    party_gender_cd_nochurn_count = nochurn.groupby(variable)[variable].count()

    party_gender_cd_churn = (party_gender_cd_churn_count / party_gender_cd_churn_count.sum()) * 100
    party_gender_cd_nochurn = (party_gender_cd_nochurn_count / party_gender_cd_nochurn_count.sum()) * 100

    difference_gender = pd.DataFrame({"churn": party_gender_cd_churn, "no churn": party_gender_cd_nochurn}).reset_index().fillna(0)
    return difference_gender

def make_butterfly_plot(df, variable, ui, i):

    # create subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=False,
                        shared_yaxes=True, horizontal_spacing=0)

    fig.append_trace(go.Bar(x=df['churn'],
                        y=df[variable], 
                        text=df["churn"].map('{:,.0f}'.format), #Display the numbers with thousands separators in hover-over tooltip 
                        textposition='inside',
                        orientation='h', 
                        width=0.7, 
                        showlegend=False, 
                        marker_color='#4472c4'), 
                        1, 1) # 1,1 represents row 1 column 1 in the plot grid

    fig.append_trace(go.Bar(x=df['no churn'],
                        y=df[variable], 
                        text=df["no churn"].map('{:,.0f}'.format),
                        textposition='inside',
                        orientation='h', 
                        width=0.7, 
                        showlegend=False, 
                        marker_color='#ed7d31'), 
                        1, 2) # 1,2 represents row 1 column 2 in the plot grid

    #fig.show()
    fig.update_xaxes(showticklabels=False,title_text="churn", row=1, col=1, autorange="reversed")
    fig.update_xaxes(showticklabels=False,title_text="no churn", row=1, col=2)

    fig.update_layout(title_text="Churn vs No churn: "+variable+" ", 
                    width=800, 
                    height=400,
                    title_x=0.9,
                    xaxis1={'side': 'top'},
                    xaxis2={'side': 'top'},)

    #fig.show()
    os.makedirs(f"static/images_differences/{ui}", exist_ok=True)
    os.makedirs(f"static/images_differences/{ui}/{i}", exist_ok=True)
    fig.write_image(f"static/images_differences/{ui}/{i}/{variable}.png")


def create_images(churn, nochurn, ui, i):
	for col_name, col_type in zip(churn.columns, churn.dtypes):
		if col_type == "object":
			difference = get_dataframe_for_plot_categoric(col_name, churn, nochurn)
			make_butterfly_plot(difference, col_name, ui, i)
		elif col_type == "float64" or col_type == "int64":
			difference = get_dataframe_for_plot_nocategoric(col_name, churn, nochurn)
			make_butterfly_plot(difference, col_name, ui, i)
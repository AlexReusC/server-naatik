# Importing the necessary libraries
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing, impute
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as plt

def get_differences_nocategoric(variable , churn, nochurn):
    churn_mean_billamount = churn[variable].mean()
    nochurn_mean_billamount = nochurn[variable].mean()
    return churn_mean_billamount, nochurn_mean_billamount

def get_differences_categoric(variable, churn, nochurn):
    count_by_type_churn = churn.groupby(variable)[variable].count()
    count_by_type_churn = ((count_by_type_churn / count_by_type_churn.sum())) * 100
    return count_by_type_churn
    count_by_type_nochurn = nochurn.groupby(variable)[variable].count()
    print(count_by_type_churn, count_by_type_nochurn)

    difference_count = count_by_type_churn.subtract(count_by_type_nochurn, fill_value=0)
    return difference_count.abs()

def calculate_difference_categoric(data, churn, nochurn):
    return round(data[0] - data[1], 2)

def calculate_difference_nocategoric(data, churn, nochurn):
    return round(data[0] - data[1], 2)

def get_dataframe_for_plot_nocategoric(variable, churn, nochurn):
    bill_amount_churn = churn[variable].describe()[["min", "25%", "50%", "75%", "max"]]
    bill_amount_no_churn = nochurn[variable].describe()[["min", "25%", "50%", "75%", "max"]]

    difference_bill = pd.DataFrame({"churn": bill_amount_churn, "no churn": bill_amount_no_churn}).reset_index()
    difference_bill.rename(columns={'index':variable}, inplace=True)
    return difference_bill

def get_dataframe_for_plot_categoric(variable, churn, nochurn):
    party_gender_cd_churn = churn.groupby(variable)[variable].count()
    party_gender_cd_nochurn = nochurn.groupby(variable)[variable].count()

    difference_gender = pd.DataFrame({"churn": party_gender_cd_churn, "no churn": party_gender_cd_nochurn}).reset_index().fillna(0)
    return difference_gender

def make_butterfly_plot(df, variable):

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
    fig.write_image("images/"+variable+".png")

def main():
    # Reading data via Pandas from CSV
    churn = pd.read_csv('./churn.csv')
    nochurn = pd.read_csv('./nochurn.csv')
    churn

    columns_churn = churn[['BILL_AMOUNT', 'COMPLAINTS', 'PARTY_GENDER_CD', 'PTY_PROFILE_SUB_TYPE', 'Years_stayed']]
    columns_nochurn = nochurn[['BILL_AMOUNT', 'COMPLAINTS', 'PARTY_GENDER_CD', 'PTY_PROFILE_SUB_TYPE', 'Years_stayed']]

   # print()
   # print("F: ",data[0] )
    diff_bill_amount = get_differences_nocategoric('BILL_AMOUNT', churn, nochurn)
    diff_complaints = get_differences_nocategoric('COMPLAINTS', churn, nochurn)

    diff_party_gender_cd = get_differences_categoric('PARTY_GENDER_CD', churn, nochurn)
    diff_pty_profile_sub = get_differences_categoric('PTY_PROFILE_SUB_TYPE', churn, nochurn)

    party_gender_cd_churn = churn.groupby("PARTY_GENDER_CD")["PARTY_GENDER_CD"].count()
    party_gender_cd_nochurn = nochurn.groupby("PARTY_GENDER_CD")["PARTY_GENDER_CD"].count()

    difference_gender = pd.DataFrame({"churn": party_gender_cd_churn, "no churn": party_gender_cd_nochurn}).reset_index().fillna(0)
    difference_gender

    df_complaints = get_dataframe_for_plot_nocategoric('COMPLAINTS', churn, nochurn)
    df_bill_amount = get_dataframe_for_plot_nocategoric('BILL_AMOUNT', churn, nochurn)
    df_years_stayed = get_dataframe_for_plot_nocategoric('Years_stayed', churn, nochurn)

    df_party_gender_cd = get_dataframe_for_plot_categoric('PARTY_GENDER_CD', churn, nochurn)
    df_pty_profile_sub_type = get_dataframe_for_plot_categoric('PTY_PROFILE_SUB_TYPE', churn, nochurn)

    make_butterfly_plot(df_party_gender_cd, 'PARTY_GENDER_CD')
    make_butterfly_plot(df_complaints, 'COMPLAINTS')
    make_butterfly_plot(df_bill_amount, 'BILL_AMOUNT')
    make_butterfly_plot(df_pty_profile_sub_type, 'PTY_PROFILE_SUB_TYPE')
    make_butterfly_plot(df_years_stayed, 'Years_stayed')

main()
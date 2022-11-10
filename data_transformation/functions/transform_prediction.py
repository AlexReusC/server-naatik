import pandas as pd
from joblib import load


# Function that transforms the dataset **for the prediction**
def transform_df_predict(df, original_name_dataset='telecom_churn_me'):

    # Loading the correspondent joblibs for the transformation
    """categorical_columns = load(f'./fragments/joblibs/{original_name_dataset}/etl/categorical_columns.joblib')
    drop_column_names = load(f'./fragments/joblibs/{original_name_dataset}/etl/drop_columns_names.joblib')
    encoder = load(f'./fragments/joblibs/{original_name_dataset}/etl/encoder.joblib')
    imp_mean = load(f'./fragments/joblibs/{original_name_dataset}/etl/imputation.joblib')
    df_mean_std = load(f'./fragments/joblibs/{original_name_dataset}/etl/mean_std.joblib')"""


    categorical_columns = load('./data_transformation/joblibs/telecom_churn_me/etl/categorical_columns.joblib')
    drop_column_names = load('./data_transformation/joblibs/telecom_churn_me/etl/drop_columns_names.joblib')
    encoder = load('./data_transformation/joblibs/telecom_churn_me/etl/encoder.joblib')
    imp_mean = load('./data_transformation/joblibs/telecom_churn_me/etl/imputation.joblib')
    df_mean_std = load('./data_transformation/joblibs/telecom_churn_me/etl/mean_std.joblib')   

    

    # Categorzing the column values with number
    categorical_columns_encoded = encoder.transform(df[categorical_columns])

    # Subsituting the numerical categorical columns in the dataset
    df[categorical_columns] = categorical_columns_encoded
    
    # Dropping the columns to drop
    df = df.drop(drop_column_names, axis=1)
    
    # Imputing the missing values of the dataset
    imputed_data = imp_mean.fit_transform(df)

    # Since the imputation returns a array, we reconvert it to a DF
    imputed_data = pd.DataFrame(imputed_data, columns=list(df.columns))
    imputed_data_names = list(imputed_data.columns)
    df[imputed_data_names] = imputed_data

    # Calculating the z-score
    z_score = df.copy()
    for column in z_score.columns:
        z_score[column] = z_score[column] - df_mean_std[column][0]
        z_score[column] = z_score[column].div(df_mean_std[column][1])

    # Updating the main variable
    df[list(z_score.columns)] = z_score

    df.to_csv(f'./transformed_new.csv', index=False)
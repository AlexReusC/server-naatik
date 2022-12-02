# Importing the necessary libraries
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing, impute
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from category_encoders import OrdinalEncoder
from joblib import dump
from imblearn.over_sampling import SMOTE
import warnings

# Function that transforms the dataset **for the modelling**
def transform_df_model(original_name_dataset, target_column_name):
    
    warnings.filterwarnings("ignore")

    df = pd.read_csv(f'./data/'+original_name_dataset+'/'+original_name_dataset+'.csv')

    # Defining the target column
    y = target_column_name

    target_column_original = df[y]

     # Calculating the total cells per client (rows per client times the columns)
    cells_per_client = len(df.columns)

    # Calculating the total non null cells per client
    non_null_cells_per_client = df.count(axis=1)



    # Calculating the total non null cells per client in terms of percentages
    percentages_non_null_cells_per_clients = (non_null_cells_per_client * 100) / (cells_per_client)

    # Defining the list of the names of the columns to drop
    drop_column_names = []

    # Obtaining the clients that have more than 75% of null values
    clients_to_drop = percentages_non_null_cells_per_clients < 25
    clients_to_drop = clients_to_drop[clients_to_drop]
    clients_to_drop = list(clients_to_drop.index)

    # There were actually no clients that had that percentage of null values
    drop_column_names.extend(clients_to_drop)

    # Calculating the total non null cells per column
    non_null_cells_per_column = df.count(axis=0)
    percentages_non_null_cells_per_column = (non_null_cells_per_column * 100) / (df.shape[0])
    
    # Obtaining the columns that have more than 65% of null values
    columns_to_drop = percentages_non_null_cells_per_column < 35
    columns_to_drop = columns_to_drop[columns_to_drop]
    columns_to_drop = list(columns_to_drop.index)
    drop_column_names.extend(columns_to_drop)

    # Storing the number of unique values of each column
    no_unique_values = df.nunique().to_frame()

    # Storing the name of columns that are full of unique values (id)
    drop_columns = no_unique_values[no_unique_values == df.shape[0]]
    drop_columns = drop_columns.dropna()
    drop_column_names.extend(drop_columns.index.tolist())
    
    # Calculating the percentiles of each column
    data_description = df.describe()
    data_description = data_description.drop(['count', 'mean', 'std', 'min', 'max'], axis = 0)

    # Storing the difference column by column of the percentiles
    data_description = data_description.diff()

    # If the difference is 0 in percentile 50 and 75, it means that the column has no variation in its values (columns full of the same value)
    percentiles = data_description[1:2] == 0.0
    percentiles = percentiles.append(data_description[2:3] == 0.0)
    for col in data_description.columns:
        if percentiles[col].all() == False:
            percentiles = percentiles.drop(col, axis=1)
    drop_column_names.extend(list(percentiles.columns))
    
    # Removing the target from the columns to drop
    y = target_column_name
    if y in drop_column_names:
        drop_column_names.remove(y)
    
    drop_column_names = list(set(drop_column_names))

    # Dropping the columns to drop
    df = df.drop(drop_column_names, axis=1)

    # Identifying the categorical columns
    categorical_columns = df.select_dtypes(include='object')
    categorical_columns = categorical_columns.columns
    categorical_columns = list(categorical_columns)

    # Removing the target from the columns to drop
    if y in categorical_columns:
        categorical_columns.remove(y)

    # Exporting the encoder
    dump(categorical_columns, f'./data_transformation/joblibs/{original_name_dataset}/etl/categorical_columns.joblib')

    # Creating a label encoder object
    encoder = OrdinalEncoder().fit(df[categorical_columns])
    
    # Categorizing the column values with numbers
    categorical_columns_encoded = encoder.transform(df[categorical_columns])

    # Exporting the encoder
    dump(encoder, f'./data_transformation/joblibs/{original_name_dataset}/etl/encoder.joblib')

    # Subsituting the numerical categorical columns in the dataset
    df[categorical_columns] = categorical_columns_encoded

    # Using the iterative imputer to estimate missing values
    imp_mean = impute.IterativeImputer()
    
    # Exporting the encoder
    dump(imp_mean, f'./data_transformation/joblibs/{original_name_dataset}/etl/imputation.joblib')

    # Imputing the missing values of the dataset
    imputed_data = imp_mean.fit_transform(df.drop([y], axis=1))
    
    # Since the imputation returns a array, we reconvert it to a DF
    imputed_data = pd.DataFrame(imputed_data, columns=list(df.drop([y], axis=1).columns))
    imputed_data_names = list(imputed_data.columns)
    df[imputed_data_names] = imputed_data

    # VIF dataframe
    vif_data = pd.DataFrame()

    # Removing the dependent variable
    x_variables = df.drop(y, axis=1)
    vif_data["x variables"] = x_variables.columns

    # Calculating the vif of the columns and dropping the high multicollinearity
    def calculate_vif(vif_data, x_variables):
        vif_data['VIF'] = [variance_inflation_factor(x_variables.values, i) for i in range(len(x_variables.columns))]
        while vif_data['VIF'].max() > 5:
            max_index = vif_data['VIF'].idxmax()
            delete_column = vif_data['x variables'].iloc[max_index]
            # Adding deleted column to the global variable
            drop_column_names.append(delete_column)
            x_variables = x_variables.drop(columns=[delete_column], axis=1)
            vif_data = vif_data.drop(index=max_index, axis=0)
            vif_data['VIF'] = [variance_inflation_factor(x_variables.values, i) for i in range(len(x_variables.columns))]
            vif_data.reset_index(inplace=True, drop=True)
        return vif_data
        

    # Storing the columns with no multicolinearity
    filtered_vif_data = calculate_vif(vif_data, x_variables)
    filtered_columns = list(filtered_vif_data['x variables'])
    filtered_columns.append(y)
    df = df[filtered_columns]

    # Obtaining the mean and standard deviation
    df_mean_std = df.describe()
    df_mean_std = df_mean_std.drop(['count', 'min', '25%', '50%', '75%', 'max'], axis = 0)

    # Exporting the data for the standardization
    dump(df_mean_std, f'./data_transformation/joblibs/{original_name_dataset}/etl/mean_std.joblib')

    # Calculating the z-score
    z_score = df.copy()
    z_score = z_score.drop([y], axis = 1)
    for column in z_score.columns:
        z_score[column] = z_score[column] - df_mean_std[column][0]
        z_score[column] = z_score[column].div(df_mean_std[column][1])

    # Updating the main variable
    df[list(z_score.columns)] = z_score

    # Defining the threshold
    threshold = 3
    # Position of the outlier
    row_postion_outlier = np.where(z_score > threshold)[0]
    row_postion_outlier = np.unique(row_postion_outlier)

    # Storing the outliers for the test
    outliers = df.iloc[row_postion_outlier]
    outliers = outliers.reset_index()

    # Removing the outliers from the main dataset
    df = df.drop(row_postion_outlier).reset_index()
    df_data = df.drop(['index'], axis=1, inplace=True)

    # Exporting the column names to drop
    dump(drop_column_names, f'./data_transformation/joblibs/{original_name_dataset}/etl/drop_columns_names.joblib')

    # Shuffling the dataset to avoid any pre-order
    df = df.sample(frac = 1).reset_index(drop = True)

    # Changing the target column to numerical
    le = preprocessing.LabelEncoder()
    le.fit(df[target_column_name])
    df[target_column_name] = le.transform(df[target_column_name])
    dump(le, f'./data_transformation/joblibs/{original_name_dataset}/etl/encoder_target.joblib')

    # Test dataset
    no_rows_test = int((df.shape[0]+outliers.shape[0])*0.3)
    test = outliers
    test = test.append(df[:no_rows_test-outliers.shape[0]]).reset_index()
    test = test.drop(['level_0', 'index'], axis=1)

    # Train dataset
    train = df[no_rows_test-outliers.shape[0]:].reset_index()
    train = train.drop(['index'], axis=1)

    # Using smote algorithm for over-sampling
    sm = SMOTE(random_state = 2)
    error_smote = False
    try:
        x_train, y_train = sm.fit_resample(train.drop([y], axis=1), train[y])
    except:
        error_smote = True

    # Calculating the general aspects of the dataset
    target_column_original = le.transform(target_column_original)
    number_churn = np.count_nonzero(target_column_original == 1)
    number_no_churn = np.count_nonzero(target_column_original == 0)
    percentage_churn = (number_churn * 100) / target_column_original.size
    percentage_no_churn = (number_no_churn * 100) / target_column_original.size

    # Saving the general aspects in a df
    d = {'total':target_column_original.size, 'number_churn': number_churn, 'number_no_churn': number_no_churn, 'percentage_churn': round(percentage_churn,2), 'percentage_no_churn': round(percentage_no_churn,2)}
    #general_aspects_original = pd.DataFrame(data=d)
    dump(d, f'./data_transformation/joblibs/{original_name_dataset}/etl/general_aspects_original.joblib')


    # Dividing the target and labels
    y_test = pd.DataFrame(test[y])
    x_test = test.drop([y], axis=1)

    # Reestructuring the train dataset
    if not error_smote:
        y_train = pd.DataFrame(y_train)

    # Exporting the test and train of the dataframes
    x_test.to_csv(f'./data/{original_name_dataset}/test/x_test.csv', index=False)
    y_test.to_csv(f'./data/{original_name_dataset}/test/y_test.csv', index=False)


    if not error_smote:
        x_train.to_csv(f'./data/{original_name_dataset}/train/x_train.csv', index=False)
        y_train.to_csv(f'./data/{original_name_dataset}/train/y_train.csv', index=False)

    train.to_csv(f'./data/{original_name_dataset}/train/original_train.csv', index=False)
    df.to_csv(f'./data/{original_name_dataset}/fully_transformed.csv', index=False)

    dump(error_smote, f'./data_transformation/joblibs/{original_name_dataset}/etl/error_smote.joblib')
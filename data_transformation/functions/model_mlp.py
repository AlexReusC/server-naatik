# Importing the necessary libraries for the data analysis and transformations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from joblib import dump
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import warnings


# Function that trains the model mlp
def train_mlp(target_column_name, original_name_dataset, smote):

    warnings.filterwarnings("ignore")

    # Obtaining the train and test dataset
    x_test = pd.read_csv(f'../data/{original_name_dataset}/test/x_test.csv')
    y_test = pd.read_csv(f'../data/{original_name_dataset}/test/y_test.csv')    

    if smote:
        x_train = pd.read_csv(f'../data/{original_name_dataset}/train/x_train.csv')
        y_train = pd.read_csv(f'../data/{original_name_dataset}/train/y_train.csv')
    else:    
        # Dividing the train dataset
        train = pd.read_csv(f'../data/{original_name_dataset}/train/original_train.csv')
        x_train = train.drop([target_column_name], axis=1)
        y_train = pd.DataFrame(train[target_column_name])

    #Initializing the MLPClassifier hyperparameters
    classifier = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(7, 5, 3), random_state=1, solver='lbfgs')

    #Fitting the training data to the network
    classifier.fit(x_train, y_train)

    #Predicting y for X_val
    y_pred_prob = classifier.predict_proba(x_test)
    y_pred = classifier.predict(x_test)

    #Creating a confusion matrix to help determinate accuracy wtih classification model
    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements
    


    #Evaluataion of the predictions against the actual observations in y_val
    cm = confusion_matrix(y_pred, y_test)

    #Storing the accuracy
    acc = round(accuracy(cm),2)
    percentage = "{:.0%}".format(acc)
    model_accuracy = (f"Accuracy of Model: {percentage}")

    # Confussion Matrix
    confussion_matrix = pd.DataFrame(cm)

    # K-Fold Cross-Validation
    def cross_validation(model, _X, _y, _cv=3):
        _scoring = ['accuracy', 'precision', 'recall', 'f1']
        results = cross_validate(estimator=model,
                                X=_X,
                                y=_y,
                                cv=_cv,
                                scoring=_scoring,
                                return_train_score=True)
        
        return {"Training Accuracy scores": results['train_accuracy'],
                "Mean Training Accuracy": results['train_accuracy'].mean()*100,
                "Training Precision scores": results['train_precision'],
                "Mean Training Precision": results['train_precision'].mean(),
                "Training Recall scores": results['train_recall'],
                "Mean Training Recall": results['train_recall'].mean(),
                "Training F1 scores": results['train_f1'],
                "Mean Training F1 Score": results['train_f1'].mean(),
                "Validation Accuracy scores": results['test_accuracy'],
                "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
                "Validation Precision scores": results['test_precision'],
                "Mean Validation Precision": results['test_precision'].mean(),
                "Validation Recall scores": results['test_recall'],
                "Mean Validation Recall": results['test_recall'].mean(),
                "Validation F1 scores": results['test_f1'],
                "Mean Validation F1 Score": results['test_f1'].mean()
                }
    
    # Grouped Bar Chart for both training and validation data
    def plot_result(x_label, y_label, plot_title, train_data, val_data):
        # Set size of plot
        plt.figure(figsize=(4,3))
        labels = ["1st Fold", "2nd Fold", "3rd Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./fragments/joblibs/{original_name_dataset}/model/mlp/k_cross_plot.png')
        plt.show()

    mlp_results = cross_validation(classifier, x_train, y_train)

    # Plot Accuracy Result
    print('Printing the K-fold Cross Validation')
    model_name = "MLP"
    plot_result(model_name,
                "Accuracy",
                "Accuracy scores in 3 Folds",
                mlp_results["Training Accuracy scores"],
                mlp_results["Validation Accuracy scores"])

    # Calculatin the MSE and accuracy in the training and test

    # Train
    y_train_predict = classifier.predict(x_train)
    y_train_true = y_train
    # MSE
    mse_train = mean_squared_error(y_train_true, y_train_predict)
    print(f'MSE Train: {mse_train}')
    # Accuracy
    acc_train = accuracy_score(y_train_true, y_train_predict, normalize=True)
    print(f'Accuracy Train: {acc_train}')

    # Test
    y_test_predict = classifier.predict(x_test)
    y_test_true = y_test
    # MSE
    mse_test = mean_squared_error(y_test_true, y_test_predict)
    print(f'MSE Test: {mse_test}')
    # Accuracy
    acc_test = accuracy_score(y_test_true, y_test_predict, normalize=True)
    print(f'Accuracy Test: {acc_test}')


    # Storing the model
    if smote:
        dump(classifier, f"./fragments/joblibs/{original_name_dataset}/model/mlp/mlp_model_smote.joblib")
    else:    
        dump(classifier, f"./fragments/joblibs/{original_name_dataset}/model/mlp/mlp_model.joblib")

    # Returning the results of the training model
    return confussion_matrix, mlp_results
# Importing the necessary libraries for the data analysis and transformations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from joblib import dump
from sklearn.model_selection import cross_validate


# Function that trains the model mlp
def train_mlp(target_column_name, original_name_dataset):

    # Obtaining the train and test dataset
    train = pd.read_csv(f'../data/{original_name_dataset}/original_train.csv')
    x_test = pd.read_csv(f'../data/{original_name_dataset}/test/x_test.csv')
    y_test = pd.read_csv(f'../data/{original_name_dataset}/test/y_test.csv')

    # Dividing the train dataset
    x_train = train.drop([target_column_name], axis=1)
    y_train = pd.DataFrame(train[target_column_name])

    #Initializing the MLPClassifier hyperparameters
    N_EPOCHS = 100
    classifier = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=N_EPOCHS,activation = 'logistic',solver='adam',random_state=1)

    #Fitting the training data to the network
    classifier.fit(x_train, y_train)

    #Predicting y for X_val
    y_pred_prob = classifier.predict_proba(x_test)
    y_pred = classifier.predict(x_test)

    #Getting the accuracy of the model
    # print(classifier.score(x_test, y_test))

    #Creating a confusion matrix to help determinate accuracy wtih classification model
    def accuracy(confusion_matrix):
        diagonal_sum = confusion_matrix.trace()
        sum_of_all_elements = confusion_matrix.sum()
        return diagonal_sum / sum_of_all_elements

    #Evaluataion of the predictions against the actual observations in y_val
    cm = confusion_matrix(y_pred, y_test)

    #Printing the accuracy
    acc = round(accuracy(cm),2)
    percentage = "{:.0%}".format(acc)
    print(f"Accuracy of Model: {percentage}")

    # Confussion Matrix
    print(pd.DataFrame(cm))

    # K-Fold Cross-Validation
    kf = KFold(n_splits=3)
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
        plt.show()

    mlp_results = cross_validation(classifier, x_train, y_train)

    # Plot Accuracy Result
    model_name = "MLP"
    plot_result(model_name,
                "Accuracy",
                "Accuracy scores in 3 Folds",
                mlp_results["Training Accuracy scores"],
                mlp_results["Validation Accuracy scores"])

    dump(classifier, f"./fragments/joblibs/{original_name_dataset}/model/classification-model.joblib")

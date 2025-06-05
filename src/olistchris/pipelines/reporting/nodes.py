import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


def create_confusion_matrix(classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series,
    X_train: pd.DataFrame, y_train: pd.Series, parameters:dict):
    """ Create a confusion matrix for classifiers

    Args:
        classifier: Trained classifier model.
        X_test: Testing data of independent features.
        y_test: Testing data for target.
        X_train: Training data of independent features.
        X_test: Testing data for target.
    Returns:
        metrics logs + metrics
    """
    data_list = [[X_test, y_test, "Test"], [X_train, y_train, "Train"]]
    plt.figure(figsize=(parameters["figsize_x"], parameters["figsize_y"]))
    for i, data in enumerate(data_list):
        pred_classes = classifier.predict(data[0])
        true_classes = data[1]

        # init confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)
        tn, fp, fn, tp = confusion_matrix(true_classes, pred_classes).ravel()

        plt.subplot(1, 2, i+1)
        sns.heatmap(cm, annot=True, fmt=parameters["fmt"], cmap=parameters["cmap"],
                    xticklabels=np.unique(true_classes),
                    yticklabels=np.unique(true_classes))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(data[2] + ' Confusion Matrix')
    return plt


def create_roc_auc(classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series,
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """ Create a roc_auc plot for classifiers

    Args:
        classifier: Trained classifier model.
        X_test: Testing data of independent features.
        y_test: Testing data for target.
        X_train: Training data of independent features.
        X_test: Testing data for target.
    Returns:
        metrics logs + metrics
    """
    plt.figure(figsize=(parameters["figsize_x"], parameters["figsize_y"]))
    lw = parameters["lw"]
    roc_colours = ['red', 'blue']
    roc_labels = ['Repeat', 'Non-Repeat']

    rocauc_lists = [[classifier.predict_proba(X_test), y_test, "Test"], [classifier.predict_proba(X_train), y_train, "Train"]]

    for i, list in enumerate(rocauc_lists):
        plt.subplot(1, 2, i + 1)
        for j in range(2):
            scores = list[0][:, j]
            fpr, tpr, _ = roc_curve(list[1] == j, scores)
            roc_auc = auc(fpr, tpr)

            # plot seperate roc auc
            plt.plot(fpr, tpr, color=roc_colours[j], lw=lw, label=f'{roc_labels[j]}(area = {roc_auc:.2f})')

        # plot grey reference lines
        plt.plot([0, 1], [0, 1], color=parameters["baseline_color"], lw=lw, linestyle=parameters["linestyle"])
        plt.plot([0, 1], [1, 1], color=parameters["baseline_color"], lw=lw, linestyle=parameters["linestyle"])

        plt.xlim([0.0, parameters["x_lim"]])
        plt.ylim([0.0, parameters["y_lim"]])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(list[2] + ' Receiver Operating Characteristics: One-vs-Rest')
        plt.legend(loc="lower right")
    return plt


def create_cont_eval(regressor, X_test: pd.DataFrame, y_test: pd.Series,
    X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    cont_list = [[X_test, y_test, "Test"], [X_train, y_train, "Train"]]
    plt.figure(figsize=(parameters["figsize_x"], parameters["figsize_y"]))
    sns.set_palette('hls')

    for i, data in enumerate(cont_list):
        y_pred = regressor.predict(data[0])

        plt.subplot(1, 2, i + 1)
        plt.scatter(y_pred, data[1], alpha=parameters["alpha"])
        plt.plot([0, 410], [0, 410], color=parameters["baseline_color"], lw=parameters["lw"])
        plt.xlabel('Predicted Value')
        plt.ylabel('True Value')
        plt.title(data[2] + " Data Regression Evaluation")
        plt.xlim(0, parameters["x_lim"])
        plt.ylim(0, parameters["y_lim"])
    return plt

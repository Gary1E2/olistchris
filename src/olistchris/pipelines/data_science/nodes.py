import logging

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingClassifier


def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data according to prediction type (classifier/regressor)
    """
    X = data[parameters["features"]]
    y = data[parameters["target"]]

    # Detect categorical columns (object type)
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if parameters["classify"] == 1:
        # Preprocessing pipeline for classifiers
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

        # Transform the data
        X_processed = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, stratify=y, test_size=parameters['test_size'],
                                                             random_state=parameters['random_state'])
    else:
        # Preprocessing pipeline for regressors
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_cols),
            ('cat', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols)
        ])

        # Transform the data
        X_processed = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2,
                                                             random_state=42)

    return X_train, X_test, y_train, y_test


def train_repeat_buyer(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> DecisionTreeClassifier:
    """Trains the repeat buyer model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for target.

    Returns:
        Trained classifier model.
    """
    classifier = GradientBoostingClassifier(n_estimators=parameters["n_estimators"], min_samples_split=parameters["min_samples_split"], min_samples_leaf=parameters["min_samples_leaf"],
                                             max_features=parameters["max_features"], max_depth=parameters["max_depth"], learning_rate=parameters["learning_rate"],
                                               random_state=parameters["random_state"])
    classifier.fit(X_train, y_train)
    return classifier


def train_freight_value(X_train: pd.DataFrame, y_train: pd.Series, parameters:dict) -> DecisionTreeClassifier:
    """Trains the freight value model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for target.

    Returns:
        Trained regressor model.
    """
    classifier = GradientBoostingRegressor(random_state=parameters["random_state"], loss=parameters["loss"], max_depth=parameters["max_depth"],
                                     max_features=parameters["max_features"], min_samples_leaf=parameters["min_samples_leaf"], min_samples_split=parameters["min_samples_split"],
                                       n_estimators=parameters["n_estimators"], learning_rate=parameters["learning_rate"])
    classifier.fit(X_train, y_train)
    return classifier


def train_delivery_time(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> DecisionTreeClassifier:
    """Trains the delivery time model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for target.

    Returns:
        Trained regressor model.
    """
    classifier = GradientBoostingRegressor(n_estimators=parameters["n_estimators"], min_samples_split=parameters["min_samples_split"], min_samples_leaf=parameters["min_samples_leaf"],
                                            max_features=parameters["max_features"], max_depth=parameters["max_depth"], learning_rate=parameters["learning_rate"],
                                            random_state=parameters["random_state"], loss=parameters["loss"])
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_classifier(
    classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series,
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained classifier model.
        X_test: Testing data of independent features.
        y_test: Testing data for target.
        X_train: Training data of independent features.
        y_train: Training data for target.
    Returns:
        metric logs + metrics
    """
    data_list = [[X_train, y_train, "Train"], [X_test, y_test, "Test"]]
    metrics_dict = {}
    logger = logging.getLogger(__name__)
    for i, data in enumerate(data_list):
        # data evaluations
        pred_classes = classifier.predict(data[0])
        true_classes = data[1]

        # init confusion matrix
        cm = confusion_matrix(true_classes, pred_classes)
        tn, fp, fn, tp = confusion_matrix(true_classes, pred_classes).ravel()

        # get results
        accuracy = accuracy_score(true_classes, pred_classes)
        precision = precision_score(true_classes, pred_classes, average='weighted')
        recall = recall_score(true_classes, pred_classes, average='weighted')
        f1 = f1_score(true_classes, pred_classes, average='weighted')
        specificity = tn / (tn+fp)


        # get roc auc
        for j in range(2):
            scores = classifier.predict_proba(X_test)[:, j]
            fpr, tpr, _ = roc_curve(y_test == j, scores)
            roc_auc = auc(fpr, tpr)

        # display results
        logger.info(data[2])
        logger.info("------------------------------------------------------------------------------")
        logger.info("Classifier accuracy: %.3f", accuracy)
        logger.info("Classifier precision: %.3f", precision)
        logger.info("Classifier recall: %.3f", recall)
        logger.info("Classifier F1: %.3f", f1)
        logger.info("Classifier specificity: %.3f", specificity)
        logger.info("Classifier ROC AUC area: %.3f", roc_auc)
        metrics_dict[data[2] + "_Accuracy"] = accuracy
        metrics_dict[data[2] + "_Precision"] = precision
        metrics_dict[data[2] + "_Recall"] = recall
        metrics_dict[data[2] + "_F1"] = f1
        metrics_dict[data[2] + "_specificity"] = specificity
        metrics_dict[data[2] + "_ROCAUC"] = roc_auc

    return metrics_dict


def evaluate_regressor(
    regressor, X_test: pd.DataFrame, y_test: pd.Series,
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained regressor model.
        X_test: Testing data of independent features.
        y_test: Testing data for target.
        X_train: Training data of independent features.
        X_test: Testing data for target.
    Returns:
        metrics logs + metrics
    """

    y_pred = regressor.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_pred) * -1
    train_mse = mean_squared_error(y_train, y_pred) * -1
    train_r2 = r2_score(y_train, y_pred)

    y_pred = regressor.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_pred) * -1
    test_mse = mean_squared_error(y_test, y_pred) * -1
    test_r2 = r2_score(y_test, y_pred)
    
    logger = logging.getLogger(__name__)
    logger.info("Train Regressor MAE: %.3f", train_mae)
    logger.info("Train Regressor MSE: %.3f", train_mse)
    logger.info("Train Regressor R^2: %.3f", train_r2)
    logger.info("Test Regressor MAE: %.3f", test_mae)
    logger.info("Test Regressor MSE: %.3f", test_mse)
    logger.info("Test Regressor R^2: %.3f", test_r2)
    
    return {"train_mae": train_mae, "train_mse": train_mse, "train_r2": train_r2, 
            "test_mae": test_mae, "test_mse": test_mse, "test_r2": test_r2}
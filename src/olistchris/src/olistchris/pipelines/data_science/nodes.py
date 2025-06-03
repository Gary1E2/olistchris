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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor


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
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, stratify=y, test_size=0.2,
                                                             random_state=42)
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


def train_repeat_buyer(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Trains the repeat buyer model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    classifier = DecisionTreeClassifier(max_depth=9, max_features=14, min_samples_leaf=2,
                                          min_samples_split=7, splitter='best', random_state=42)
    classifier.fit(X_train, y_train)
    return classifier


def train_freight_value(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Trains the freight value model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    classifier = GradientBoostingRegressor(random_state=42, loss='absolute_error', max_depth=8,
                                     max_features=7, min_samples_leaf=3, min_samples_split=2,
                                       n_estimators=86, learning_rate=0.22)
    classifier.fit(X_train, y_train)
    return classifier


def train_delivery_time(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Trains the delivery time model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    classifier = GradientBoostingRegressor(n_estimators=50, min_samples_split=3, min_samples_leaf=2,
                                            max_features=3, max_depth=5, learning_rate=0.3,
                                            random_state=42, loss='absolute_error')
    classifier.fit(X_train, y_train)
    return classifier


def evaluate_classifier(
    classifier: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = classifier.predict(X_test)

    # init confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # custom metric calculations
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    specificity = tn / (tn+fp)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient test acc of %.3f on test data.", accuracy)
    return {"accuracy": accuracy, "precision": precision , "recall": recall,
             "f1": f1, "specificity": specificity}


def evaluate_regressor(
    regressor: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series,
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for target.
        X_train: Training data of independent features.
        X_test: Testing data for target.
    """
    test_mae = mean_absolute_error(y_test, regressor.predict((X_test))) * -1
    train_mae = mean_absolute_error(y_train, regressor.predict((X_train))) * -1
    test_mse = mean_squared_error(y_test, regressor.predict((X_test))) * -1
    train_mse = mean_squared_error(y_train, regressor.predict((X_train))) * -1
    test_r2 = r2_score(y_test, regressor.predict((X_test)))
    train_r2 = r2_score(y_train, regressor.predict((X_train)))
    
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient test mae of %.3f on test data.", test_mae)
    return {"test_mae": test_mae, "train_mae": train_mae, "test_mse": test_mse, "train_mse": train_mse,
            "test_r2": test_r2, "train_r2": train_r2}
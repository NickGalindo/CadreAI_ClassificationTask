from typing import List, Dict, Tuple
from manager.load_config import CONFIG
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from pprint import pprint

def loadData(file_path: str) -> pd.DataFrame:
    """
    Load csv data from a file

    :param file_path: the path to the csv file to load as a dataframe
    :return: The dataframe loaded from the csv
    """
    return pd.read_csv(file_path)

def preprocessData(df: pd.DataFrame, property_type_as_ordinal: bool =False) -> Tuple[np.ndarray, np.ndarray]:
    """
    preprocess the data in the df to prepare it for inference

    :param df: the dataframe where the csv data has been loaded
    :param property_type_as_ordinal: optional parameter passed for inference. It helps test if encoding property_type as an ordinal provides better results over one-hot encoding it
    :return: Returns a tuple where the first element are the features and the second element are the target variables
    """
    bool_cols: List = ['region_East', 'region_North', 'region_South', 'region_West', 'loan_status']

    df = df.drop_duplicates() # Drop duplicates even though it has none
    df = df.dropna(subset=["application_id"]) # drop any where the application_id is missing although there are none
    df = df.drop(columns=["application_id"]) # not an inference variable


    df["loan_status"] = df["loan_status"] == "approved"
    df = pd.get_dummies(df, columns=["region"])

    if property_type_as_ordinal:
        PROPERTY_TYPE_ORDER: Dict = {
            "condo": 1,
            "townhouse": 2,
            "multi_unit": 3,
            "single_family": 4
        }

        df["property_type"] = df["property_type"].map(lambda x: PROPERTY_TYPE_ORDER[x] if x in PROPERTY_TYPE_ORDER else np.nan)
    else:
        df = pd.get_dummies(df, columns=["property_type"])
        bool_cols += ['property_type_condo', 'property_type_multi_unit', 'property_type_single_family', 'property_type_townhouse']

    for col in bool_cols:
        df[col] = df[col].astype(int)

    # cast all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1) # set missing vlues to -1, makes it clear to the model that it's a missing value. Other techniques could apply

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    return X.to_numpy(), y.to_numpy()

def trainAndEvaluate(X: np.ndarray, y: np.ndarray) -> None:
    """
    train and evaluate the models with the given features and target vectors

    :param X: feature matrix
    :param y: target vector
    :return: nothing
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model: XGBClassifier = XGBClassifier(objective="binary:logistic", n_estimators=1000, learning_rate=1, max_depth=100, random_state=42)

    reports: Dict[float, List] = {}

    lr_range: np.ndarray = np.arange(0.01, 0.5, 0.05)

    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    
    for idx, weight in enumerate(np.arange(0, 4.5, 0.5)):
        train: List = []
        test: List = []
        reports_per_weight: List = []

        fig2, axs2 = plt.subplots(nrows=2, ncols=5, figsize=(10, 10))

        for jdx, lr in enumerate(lr_range):
            model = XGBClassifier(eta=lr, reg_lambda=1, min_child_weight=weight)
            model.fit(X_train, y_train)
            train.append(model.score(X_train, y_train))
            test.append(model.score(X_test, y_test))

            y_pred = model.predict(X_test)
            reports_per_weight.append(classification_report(y_test, y_pred))

            y_pred_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_prob)
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

            axs2[jdx//5][jdx%5].plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {auc:.2f})')
            axs2[jdx//5][jdx%5].plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (no discrimination)
            axs2[jdx//5][jdx%5].set_xlabel('False Positive Rate')
            axs2[jdx//5][jdx%5].set_ylabel('True Positive Rate')
            axs2[jdx//5][jdx%5].set_title(f"ROC Curve (Min child weight: {weight})", size=16)
            axs2[jdx//5][jdx%5].legend(loc='best')
            axs2[jdx//5][jdx%5].grid(True)

        fig2.subplots_adjust(hspace=1)
        fig2.tight_layout(pad=3.0)
        fig2.savefig(f"./data/images/roc_plot_min_child_weight_{weight}.png")
        plt.close(fig2)
            

        reports[float(weight)] = reports_per_weight

        axs[idx//3][idx%3].plot(lr_range, train, c='orange', label='Training')
        axs[idx//3][idx%3].plot(lr_range, test, c='m', label='Testing')
        axs[idx//3][idx%3].set_xlabel('Learning rate')
        axs[idx//3][idx%3].set_xticks(lr_range)
        axs[idx//3][idx%3].set_ylabel('Accuracy score')
        axs[idx//3][idx%3].set_ylim(0.6, 1)
        axs[idx//3][idx%3].legend(prop={'size': 12}, loc=3)
        axs[idx//3][idx%3].set_title(f"Min child weight: {weight}", size=16)
        axs[idx//3][idx%3].grid(True)
    
    fig.subplots_adjust(hspace=1)
    fig.tight_layout(pad=3.0)
    fig.savefig(f"./data/images/accuracy_plots.png")
    plt.close(fig)

    pprint(reports)


def main():
    df: pd.DataFrame = loadData("./data/mortgage_training_data.csv")
    X, y = preprocessData(df)
    trainAndEvaluate(X, y)

    return

if __name__ == '__main__':
    main()

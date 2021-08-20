import h1st as h1
import sys

import config
import os
import boto3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import sklearn


class WineQualityModel(h1.Model):
    def __init__(self):
        super().__init__()
        self.model = None
        self.features = [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ]
        self.data_dir = config.DATA_PATH
        self.test_size = 0.2

    def load_data(self):
        filename = "s3://arimo-pana-cyber/explain_data/winequality_red.csv"
        df = pd.read_csv(filename)
        df["quality"] = df["quality"].astype(int)
        self.data = df

    def explore(self):
        self.data["quality"].hist()
        plt.title("Wind Quality Rating")

    def prep_data(self):
        """
        Prepare data for modelling
        :param loaded_data: data return from load_data method
        :returns: dictionary contains train data and validation data
        """

        Y = self.data["quality"]
        X = self.data[self.features]
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_size
        )

        return {
            "train_df": X_train,
            "val_df": X_test,
            "train_labels": Y_train,
            "val_labels": Y_test,
        }

    def train(self, prepared_data):
        X_train, Y_train = prepared_data["train_df"], prepared_data["train_labels"]
        model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
        model.fit(X_train, Y_train)
        self.model = model

    def evaluate(self, prepared_data):
        X_test, y_true = prepared_data["val_df"], prepared_data["val_labels"]
        y_pred = self.model.predict(X_test)
        self.metrics = {
            "mae": sklearn.metrics.mean_absolute_error(y_true, y_pred),
        }
        print(self.metrics)

    def predict(self, filename):
        df = pd.read_csv(filename)
        df["quality"] = df["quality"].astype(int)
        input_data = df[self.features]
        return self.model.predict(input_data)

    def explain(self, prepared_data):
        h1.SHAPExplainer(self, prepared_data)
        h1.LIMEExplainer(self, prepared_data)

import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from nbsvm_classifier import NBSVMClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class TrainClassifier:
    def __init__(self, field="Summary", vec_type="tfidf"):
        self.RANDOM_STATE = 42
        self.field = field
        self.vec_type = vec_type

    def get_data(self, path):
        df = pd.read_csv(path)
        df = df.drop_duplicates()
        df = df.dropna(subset=[self.field])
        X, y = df[self.field], df["Class"]
        return X, y

    def get_splits(self, X, y):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.RANDOM_STATE,
            stratify=y,
        )
        return X_train, X_valid, y_train, y_valid

    def print_configs(self):
        print(f"FIELD: {self.field}")
        print(f"VECTORIZER: {self.vec_type.upper()}", end="\n\n")

    def print_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        print(f"ACCURACY: {acc:.3f}")
        print(f"MCC: {mcc:.3f}")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"TP: {tp}\tTN: {tn}\tFP: {fp}\tFN: {fn}", end="\n\n")
        return acc, mcc, tp, tn, fp, fn

    def get_count_vectors(self, X_train, X_valid):
        vec = CountVectorizer()
        train_term_doc = vec.fit_transform(X_train)
        valid_term_doc = vec.transform(X_valid)
        joblib.dump(vec, f"../pickles/count_vectorizer_{self.field}.joblib")
        return train_term_doc, valid_term_doc

    def get_tfidf_vectors(self, X_train, X_valid):
        vec = TfidfVectorizer()
        train_term_doc = vec.fit_transform(X_train)
        valid_term_doc = vec.transform(X_valid)
        joblib.dump(vec, f"../pickles/tfidf_vectorizer_{self.field}.joblib")
        return train_term_doc, valid_term_doc

    def train(self, path):
        accs, mccs, tps, tns, fps, fns = [], [], [], [], [], []

        # load data
        X, y = self.get_data(path)

        # get splits
        X_train, X_valid, y_train, y_valid = self.get_splits(X, y)

        # get train & valid vectors
        if self.vec_type == "tfidf":
            train_term_doc, valid_term_doc = self.get_tfidf_vectors(X_train, X_valid)
        elif self.vec_type == "count":
            train_term_doc, valid_term_doc = self.get_count_vectors(X_train, X_valid)

        # iterate over each model
        for model_name in ["naive_bayes", "nbsvm", "logistic_regression"]:
            print("*" * 60)
            print(" " * 25 + f"{model_name.upper()}")
            print("*" * 60)
            # create an object
            if model_name == "naive_bayes":
                model = MultinomialNB()
            elif model_name == "nbsvm":
                model = NBSVMClassifier()
            elif model_name == "logistic_regression":
                model = LogisticRegression()

            # fit the model on training data
            model.fit(train_term_doc, y_train)

            # predict on train & validation
            train_preds = model.predict(train_term_doc)
            valid_preds = model.predict(valid_term_doc)

            # print & store training metrics
            print(f"TRAINING METRICS")
            acc, mcc, tp, tn, fp, fn = self.print_metrics(y_train, train_preds)
            accs.append(acc)
            mccs.append(mcc)
            tps.append(tp)
            tns.append(tn), fps.append(fp), fns.append(fn)

            # print & store validation metrics
            print(f"VALIDATION METRICS")
            acc, mcc, tp, tn, fp, fn = self.print_metrics(y_valid, valid_preds)
            accs.append(acc)
            mccs.append(mcc)
            tps.append(tp)
            tns.append(tn), fps.append(fp), fns.append(fn)
            print("\n\n")

            # save model
            joblib.dump(
                model, f"../models/{model_name}_{self.field}_{self.vec_type}.joblib"
            )
        index = pd.MultiIndex.from_product(
            [["naive_bayes", "nbsvm", "logistic_regression"], ["Train", "Valid"]],
            names=["Split", "Model"],
        )
        metric_df = pd.DataFrame.from_dict(
            {"ACC": accs, "MCC": mccs, "TP": tps, "TN": tns, "FP": fps, "FN": fns},
        )
        metric_df.index = index
        print(metric_df.to_markdown())


if __name__ == "__main__":
    path = "../data/processed/processed_food_reviews.csv"
    for vec_type in ["count", "tfidf"]:
        for field in ["Summary", "Text"]:
            classifier = TrainClassifier(field=field, vec_type=vec_type)
            classifier.print_configs()
            classifier.train(path)
            print("-" * 80, end="\n")
        print(">" * 80, end="\n")

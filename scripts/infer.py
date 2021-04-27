import joblib
import fire
import numpy as np
import pandas as pd

from process import Processor


class Infer:
    def __init__(self):
        self.processor = Processor()
        self.summary_vectorizer = joblib.load(
            "../pickles/tfidf_vectorizer_Summary.joblib"
        )
        self.summary_model = joblib.load(
            "../models/logistic_regression_Summary_tfidf.joblib"
        )
        self.text_vectorizer = joblib.load("../pickles/tfidf_vectorizer_Text.joblib")
        self.text_model = joblib.load("../models/logistic_regression_Text_tfidf.joblib")

    def predict_summary(self, text):
        text = self.processor.process_all(text)
        vectors = self.summary_vectorizer.transform(np.array([text]))
        prediction = self.summary_model.predict(vectors)[0]
        prob_score = self.summary_model.predict_proba(vectors)[0]
        print(f"Prediction: {prediction}")
        print(f"Probability Scores: {prob_score}")
        # return prediction

    def predict_review(self, text):
        text = self.processor.process_all(text)
        vectors = self.text_vectorizer.transform([text])
        prediction = self.text_model.predict(vectors)[0]
        prob_score = self.text_model.predict_proba(vectors)[0]
        print(f"Prediction: {prediction}")
        print(f"Probability Scores: {prob_score}")
        # return prediction


if __name__ == "__main__":
    fire.Fire(Infer)
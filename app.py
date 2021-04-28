import pickle
import re
import string

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from nltk.stem import WordNetLemmatizer


class Processor:
    def __init__(self):
        self.STOPWORDS = [
            "an",
            "re",
            "have",
            "any",
            "then",
            "further",
            "here",
            "about",
            "were",
            "in",
            "just",
            "no",
            "below",
            "needn",
            "not",
            "so",
            "myself",
            "which",
            "more",
            "theirs",
            "hadn't",
            "themselves",
            "our",
            "isn",
            "wasn",
            "it's",
            "s",
            "some",
            "won't",
            "down",
            "wouldn",
            "hers",
            "there",
            "wasn't",
            "y",
            "will",
            "him",
            "d",
            "all",
            "ll",
            "after",
            "very",
            "when",
            "doesn",
            "during",
            "by",
            "above",
            "those",
            "weren't",
            "are",
            "ain",
            "isn't",
            "yourselves",
            "is",
            "t",
            "you'll",
            "as",
            "each",
            "both",
            "herself",
            "your",
            "does",
            "same",
            "once",
            "other",
            "you've",
            "over",
            "off",
            "hasn",
            "doesn't",
            "been",
            "haven't",
            "me",
            "its",
            "haven",
            "couldn't",
            "ourselves",
            "if",
            "has",
            "she's",
            "he",
            "own",
            "between",
            "i",
            "had",
            "only",
            "too",
            "don",
            "to",
            "what",
            "she",
            "until",
            "on",
            "should've",
            "why",
            "didn't",
            "am",
            "shan",
            "again",
            "ve",
            "but",
            "you're",
            "you",
            "mustn't",
            "it",
            "with",
            "into",
            "before",
            "that",
            "out",
            "o",
            "under",
            "shouldn't",
            "a",
            "that'll",
            "don't",
            "can",
            "up",
            "doing",
            "itself",
            "they",
            "should",
            "shan't",
            "do",
            "while",
            "from",
            "be",
            "or",
            "such",
            "being",
            "how",
            "against",
            "few",
            "for",
            "won",
            "having",
            "nor",
            "ma",
            "didn",
            "now",
            "of",
            "aren't",
            "wouldn't",
            "weren",
            "because",
            "my",
            "than",
            "m",
            "his",
            "this",
            "yourself",
            "these",
            "mightn",
            "through",
            "who",
            "mightn't",
            "whom",
            "yours",
            "we",
            "ours",
            "shouldn",
            "her",
            "at",
            "their",
            "mustn",
            "aren",
            "where",
            "was",
            "the",
            "needn't",
            "them",
            "hadn",
            "most",
            "did",
            "you'd",
            "hasn't",
            "couldn",
            "himself",
            "and",
        ]
        self.PUNCT_TO_REMOVE = "\"#$%&'*+,-./<=>?@[\\]^_`{|}~"
        self.lemmatizer = WordNetLemmatizer()
        self.url_pattern = re.compile(
            r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
        )
        self.RAREWORDS = pickle.load(open("pickles/rarewords.pkl", "rb"))

    def process_all(self, text):
        text = self.lower_case(text)
        text = self.remove_puncutations(text)
        text = self.remove_stopwords(text)
        text = self.remove_rarewords(text)
        text = self.lemmatize_words(text)
        text = self.remove_urls(text)
        return text

    def lower_case(self, text):
        return text.strip().lower()

    def remove_stopwords(self, text):
        return " ".join(
            [word for word in str(text).split() if word not in self.STOPWORDS]
        )

    def remove_puncutations(self, text):
        return text.translate(str.maketrans("", "", self.PUNCT_TO_REMOVE))

    def lemmatize_words(self, text):
        return " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def remove_rarewords(self, text):
        return " ".join(
            [word for word in str(text).split() if word not in self.RAREWORDS]
        )

    def remove_urls(self, text):
        return self.url_pattern.sub(r"", text)


class Infer:
    def __init__(self):
        self.processor = Processor()
        self.summary_vectorizer = joblib.load("pickles/tfidf_vectorizer_Summary.joblib")
        self.summary_model = joblib.load(
            "models/logistic_regression_Summary_tfidf.joblib"
        )
        self.text_vectorizer = joblib.load("pickles/tfidf_vectorizer_Text.joblib")
        self.text_model = joblib.load("models/logistic_regression_Text_tfidf.joblib")

    def predict_summary(self, text):
        text = self.processor.process_all(text)
        vectors = self.summary_vectorizer.transform(np.array([text]))
        prediction = self.summary_model.predict(vectors)[0]
        prob_score = self.summary_model.predict_proba(vectors)[0]
        print(f"Prediction: {prediction}")
        print(f"Probability Scores: {prob_score}")
        return prediction, prob_score

    def predict_review(self, text):
        text = self.processor.process_all(text)
        vectors = self.text_vectorizer.transform([text])
        prediction = self.text_model.predict(vectors)[0]
        prob_score = self.text_model.predict_proba(vectors)[0]
        print(f"Prediction: {prediction}")
        print(f"Probability Scores: {prob_score}")
        return prediction, prob_score


if __name__ == "__main__":
    # inference object
    infer = Infer()
    # get summary
    summary_slot = st.empty()
    summary_query = st.text_input("Summary", "")
    # predict summary
    if st.button("Predict Summary"):
        if summary_query:
            summary_pred, summary_score = infer.predict_summary(summary_query)
            st.success(summary_pred)
            st.bar_chart(
                {"Negative": [summary_score[0]], "Positive": [summary_score[1]]}
            )
        else:
            st.warning("Summary cannot be blank!")

    # get review
    review_slot = st.empty()
    review_query = st.text_input("Review", "")
    # predict review
    if st.button("Predict Review"):
        if review_query:
            review_pred, review_score = infer.predict_review(review_query)
            st.success(review_pred)
            st.bar_chart({"Negative": [review_score[0]], "Positive": [review_score[1]]})
        else:
            st.warning("Review cannot be blank!")

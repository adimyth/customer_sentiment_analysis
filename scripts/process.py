import pickle
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Processor:
    def __init__(self):
        self.STOPWORDS = set(stopwords.words("english"))
        self.PUNCT_TO_REMOVE = "\"#$%&'*+,-./<=>?@[\\]^_`{|}~"
        self.lemmatizer = WordNetLemmatizer()
        self.url_pattern = re.compile(
            r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
        )
        self.RAREWORDS = pickle.load(open("../pickles/rarewords.pkl", "rb"))

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

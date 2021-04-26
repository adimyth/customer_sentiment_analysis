# Customer Sentiment Analysis
## Problem Description
> "Create a model that predicts scores from text reviews"

### Business Case
The web contains enormous amounts of user generated data. One huge source of such information can be found in the myriads of customer written complaints and reviews.
The aim of this data challenge is to show that it is possible to extract useful insights from text and that there is value in text data. Can you convince the C-suite of an online food seller company that there is value in working with the text data? In a first meeting you want to show them, that a machine learning model can be used to distinguish between different levels of consumer satisfaction

### The situation (detailed)
The food retailer wants to understand their users as well as users' feelings about the products they are selling.
How could a model as described above help in this situation? Make assumptions or educated guesses if some details are missing that you would like to use for your argument.

## Data
### Predictive Fields
...

### Target Field
...

## Preprocessing
* Lower Casing
* Remove Punctuations
* Stopwords Removal
* Remove Frequent Words
* Remove Rare Words
* Lemmatization
* Remove URLs
* Scores to Class conversion

## Training
...

## Evaluation
...

## Baselines
### Random Baseline
Randomly predict a class. Since, it's a binary classification -
```python
Accuracy = 50%
```

### Most Frequent Class
Predict the most frequent class all times. Since, `positive` class is the frequent class. Baseline accuracy -
```
Accuracy = # positive samples / # total samples
Accuracy = 309000/380000 = ~0.80
```

## Machine Learning Models
Train different ML models on the summary & text vectors. We are using `TfIdfVectorizer` & `CountVectorizer` to convert text & summary into corresponding vectors. Here, we compare the following models -
* Multinomial Naive Bayes
* NBSVM
* Logisitic Regression

### Count Vectorizer
#### Summary
|                                  |      ACC |      MCC |    TP |     TN |    FP |    FN |
|:---------------------------------|---------:|---------:|------:|-------:|------:|------:|
| ('naive_bayes', 'Train')         | 0.874254 | 0.607949 | 41342 | 234346 | 11644 | 28009 |
| ('naive_bayes', 'Valid')         | 0.860305 | 0.558356 |  9382 |  58441 |  3057 |  7956 |
| ('nbsvm', 'Train')               | 0.780076 | 0        |     0 | 245990 |     0 | 69351 |
| ('nbsvm', 'Valid')               | 0.780075 | 0        |     0 |  61498 |     0 | 17338 |
| ('logistic_regression', 'Train') | 0.878988 | 0.61969  | 40136 | 237045 |  8945 | 29215 |
| ('logistic_regression', 'Valid') | 0.866558 | 0.576958 |  9387 |  58929 |  2569 |  7951 |

#### Text
|                                  |      ACC |      MCC |    TP |     TN |    FP |    FN |
|:---------------------------------|---------:|---------:|------:|-------:|------:|------:|
| ('naive_bayes', 'Train')         | 0.887138 | 0.662039 | 48713 | 231516 | 14638 | 21013 |
| ('naive_bayes', 'Valid')         | 0.869229 | 0.601085 | 10935 |  57708 |  3830 |  6497 |
| ('nbsvm', 'Train')               | 0.779264 | 0        |     0 | 246154 |     0 | 69726 |
| ('nbsvm', 'Valid')               | 0.779258 | 0        |     0 |  61538 |     0 | 17432 |
| ('logistic_regression', 'Train') | 0.913081 | 0.735889 | 50031 | 238393 |  7761 | 19695 |
| ('logistic_regression', 'Valid') | 0.878612 | 0.627379 | 11045 |  58339 |  3199 |  6387 |


### TFIDF Vectorizer
#### Summary
|                                  |      ACC |      MCC |    TP |     TN |   FP |    FN |
|:---------------------------------|---------:|---------:|------:|-------:|-----:|------:|
| ('naive_bayes', 'Train')         | 0.868736 | 0.579055 | 33044 | 240904 | 5086 | 36307 |
| ('naive_bayes', 'Valid')         | 0.854445 | 0.52485  |  7360 |  60001 | 1497 |  9978 |
| ('nbsvm', 'Train')               | 0.780076 | 0        |     0 | 245990 |    0 | 69351 |
| ('nbsvm', 'Valid')               | 0.780075 | 0        |     0 |  61498 |    0 | 17338 |
| ('logistic_regression', 'Train') | 0.875693 | 0.606949 | 38465 | 237677 | 8313 | 30886 |
| ('logistic_regression', 'Valid') | 0.865988 | 0.573556 |  9177 |  59094 | 2404 |  8161 |

#### Text
|                                  |      ACC |      MCC |    TP |     TN |   FP |    FN |
|:---------------------------------|---------:|---------:|------:|-------:|-----:|------:|
| ('naive_bayes', 'Train')         | 0.800481 | 0.274283 |  6944 | 245912 |  242 | 62782 |
| ('naive_bayes', 'Valid')         | 0.791858 | 0.209404 |  1053 |  61480 |   58 | 16379 |
| ('nbsvm', 'Train')               | 0.779264 | 0        |     0 | 246154 |    0 | 69726 |
| ('nbsvm', 'Valid')               | 0.779258 | 0        |     0 |  61538 |    0 | 17432 |
| ('logistic_regression', 'Train') | 0.896765 | 0.68293  | 46200 | 237070 | 9084 | 23526 |
| ('logistic_regression', 'Valid') | 0.884804 | 0.645047 | 11085 |  58788 | 2750 |  6347 |

## Getting Started
All the experiments are run on `python 3.7.0`.

1. Clone the repository
2. If you do not have python3.7 installed. Run the below steps for easy installation using [asdf](https://asdf-vm.com/). *asdf* allows us to manage multiple runtime versions such for different languages such as `nvm`, `rbenv`, `pyenv`, etc using a CLI tool
	* Install asdf using this [guide](https://asdf-vm.com/#/core-manage-asdf-vm?id=install)
	* Now install `python3.7.0`
	```bash
	asdf plugin add python
	asdf install python 3.7.0
	asdf local python 3.7.0
	```
	* Check the set python version
	```bash
	asdf current python
	```
3. Install poetry. [Poetry](https://python-poetry.org/docs/) is a python dependency management & packaging tool. Allows us to declare project libraries dependency & manage them
	```bash
	asdf plugin add poetry
	asdf install poetry latest
	asdf local poetry 1.1.1
	```
4. Install all dependencies
	```bash
	poetry install
	```

## Project Structure
```bash
.
├── data
│   ├── intermediate
│   ├── processed
│   └── raw
├── notebooks
├── pickles
├── poetry.lock
├── pyproject.toml
└── scripts
```
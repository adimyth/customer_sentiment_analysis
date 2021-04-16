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
...

## Training
...

## Evaluation
...

## Baselines
### Random Baseline
...

### Most Frequent Class
...

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
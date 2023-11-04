# Project: Naïve Bayes for Text Classification

## Introduction

This project focuses on applying Naïve Bayes for text classification on a toxic comment dataset. The objective is to classify text as either toxic (1) or not toxic (0).

## Implementation

### Python Packages

- We use the scikit-learn package for implementing Naïve Bayes, specifically the `sklearn.naive_bayes` module, which provides efficient tools for text classification. You can find more information about scikit-learn Naïve Bayes [here](https://scikit-learn.org/stable/modules/naive_bayes.html).

### Dataset

- The dataset used for this project is a toxic comment dataset. It contains comments that may be offensive or harmful, so caution is advised when handling the data.
- The dataset consists of training and test sets, which can be obtained [here](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).
- The training set (`train.csv`) is used for training the Naïve Bayes model, while the test set (`test.csv`) and labels (`test_labels.csv`) are used for testing and analyzing the models.

### Naïve Bayes Model Tasks

We have implemented and tested the Naïve Bayes model, focusing on the following key tasks:

1. **Training a Naïve Bayes Model:** We trained the Naïve Bayes model on the training text using CountVectorizer from scikit-learn. This model is designed to predict whether text is toxic or not toxic.

2. **Testing the Naïve Bayes Model:** We tested the trained Naïve Bayes model on the test set, producing predictions for all test texts.

3. **Analyzing Model Accuracy:** We analyzed the accuracy of our Naïve Bayes model and made efforts to improve it. This involved experimenting with different approaches, such as adding additional features, refining text preprocessing, and adjusting hyperparameters of the Naïve Bayes model for training. We took care to avoid overfitting, as overfit models can lead to poor generalization.

4. **Comparing Toxic vs. Non-Toxic Texts:** We compared and contrasted how the model performed on different labeled data, specifically toxic vs. non-toxic texts.

5. **Report and Observations:** Our findings and attempts to improve model accuracy are documented in a report alongside the code. We provided observations even if certain changes led to a drop in accuracy, explaining the rationale behind any changes.

## Conclusion

This project allowed us to apply the Naïve Bayes algorithm to text classification. We gained insights into the challenges and strategies for improving model accuracy in the context of toxic comment classification.

This project leverages the scikit-learn package for Naïve Bayes implementation and the toxic comment dataset provided by Kaggle.

## Author

- Suchir Naik

Feel free to explore and modify the code to enhance your understanding and to improve the performance of the Naïve Bayes model.


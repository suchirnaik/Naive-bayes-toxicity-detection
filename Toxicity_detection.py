#To run this program please save the test.csv, train.csv and test_labels.csv in the same location as this python file
#I have given the relative path for all test, train and test_labels csv
#If you would like to change the probability threshold please go to line 64 to change the threshold value
#When this program is run a csv file called predictions.csv will be saved to the folder location where this python file is saved.
#The predictions.csv contains the probability of that text being toxic and class prediction(toxic, non toxic) 
#There is another file called final dataframe which is also outputed. This dataframe is the final dataframe with the -1 labels removed
#The final dataframe contains the columns comment_text toxic_prob_MNB toxic_prob_CNB toxic_pred_MNB toxic_pred_CNB toxic(this is the actual label from test_labels.csv)

import pandas as pd
import re
# Load train data
train_data = pd.read_csv('train.csv')

# Load test data and labels
test_data = pd.read_csv('test.csv')
test_labels = pd.read_csv('test_labels.csv')
train_data_toxic = train_data[train_data['toxic']==1]
train_data_non_toxic = train_data[train_data['toxic']==0]
def preprocess_text(text):
    # Remove unnecessary characters(special characters) and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    return text
#applying pre processing function on train data
train_text = train_data['comment_text'].apply(preprocess_text)
train_labels = train_data['toxic']
#Removing rows where the toxic labels are -1
test_data = test_data[test_labels['toxic'] != -1]
test_labels = test_labels[test_labels['toxic'] != -1]
#apply preprocessing function on test data
test_text = test_data['comment_text'].apply(preprocess_text)
test_labels = test_labels['toxic']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
#Train function
def train_NB_model(path_to_train_file):
    # Load train data
    train_data = pd.read_csv(path_to_train_file)
    train_text = train_data['comment_text'].apply(preprocess_text)
    train_labels = train_data['toxic']

    # Convert text to bag-of-words representation
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_text)
    
    # Train Gaussian Naïve Bayes model
    #gnb = GaussianNB()
    #gnb.fit(X_train.toarray(), train_labels)

    #Train Multinomial Naïve Bayes model
    mnb = MultinomialNB()
    mnb.fit(X_train, train_labels)
    #Train Complement Naive Bayes
    cnb = ComplementNB()
    cnb.fit(X_train, train_labels)

    return mnb,cnb,vectorizer
#Initializing the models and vectorizer for the test function
MNB_model,CNB_model,vectorizer = train_NB_model("train.csv")
#Test function
#setting the probability threshold to 0.5
def test_NB_model(path_to_test_file, MNB_model, CNB_model, vectorizer, threshold=0.5):
    # Load test data
    test_data = pd.read_csv(path_to_test_file)
    test_text = test_data['comment_text'].apply(preprocess_text)

    # Convert text to bag-of-words representation
    X_test = vectorizer.transform(test_text)

    # probability of that text being toxic
    test_data['toxic_prob_MNB'] = MNB_model.predict_proba(X_test)[:, 1]
    test_data['toxic_prob_CNB'] = CNB_model.predict_proba(X_test)[:, 1]
    
    #Making class prediction (toxic, not toxic)
    # Set the predicted labels based on the specified threshold
    test_data['toxic_pred_MNB'] = (MNB_model.predict_proba(X_test)[:, 1] > threshold).astype(int)
    test_data['toxic_pred_CNB'] = (CNB_model.predict_proba(X_test)[:, 1] > threshold).astype(int)
    #Outputtig the file to csv
    test_data.to_csv('predictions.csv', index=False)
    
    return test_data

#creating a dataframe called predicted_labels
predicted_labels = test_NB_model("test.csv", MNB_model,CNB_model,vectorizer)



# Read in test labels
test_labels = pd.read_csv("test_labels.csv")
#Merging test labels and predicted labels
merged_labels = pd.merge(test_labels, predicted_labels, on="id")
#Selecting only required columns
merged_labels_1 = merged_labels[['comment_text','toxic_prob_MNB','toxic_prob_CNB','toxic_pred_MNB','toxic_pred_CNB','toxic']]
#Removing toxic lables equal to -1
merged_labels_1 = merged_labels_1[merged_labels_1['toxic'] != -1]
#Predicting the accuracy for Multinomial Naïve Bayes model and Complement Naive Bayes model for the whole dataset
accuracy = (merged_labels_1["toxic"] == merged_labels_1["toxic_pred_MNB"]).mean()
print("Accuracy of Multinomial Naïve Bayes model :", accuracy)
accuracy = (merged_labels_1["toxic"] == merged_labels_1["toxic_pred_CNB"]).mean()
print("Accuracy of Complement Naive Bayes model :", accuracy)

# Calculate accuracy on toxic texts for Multinomial Naïve Bayes model
toxic_accuracy_MNB = (merged_labels_1["toxic"] == merged_labels_1["toxic_pred_MNB"])[merged_labels_1["toxic"] == 1].mean()
print("Toxic Accuracy for Multinomial Naïve Bayes model :", toxic_accuracy_MNB)

# Calculate accuracy on non-toxic texts for Multinomial Naïve Bayes model
nontoxic_accuracy_MNB = (merged_labels_1["toxic"] == merged_labels_1["toxic_pred_MNB"])[merged_labels_1["toxic"] == 0].mean()
print("Non-Toxic Accuracy Multinomial Naïve Bayes model:", nontoxic_accuracy_MNB)

# Calculate accuracy on toxic texts for Complement Naive Bayes model
toxic_accuracy_CNB = (merged_labels_1["toxic"] == merged_labels_1["toxic_pred_CNB"])[merged_labels_1["toxic"] == 1].mean()
print("Toxic Accuracy for Complement Naive Bayes model :", toxic_accuracy_CNB)

# Calculate accuracy on non-toxic texts for Complement Naive Bayes model
nontoxic_accuracy_CNB = (merged_labels_1["toxic"] == merged_labels_1["toxic_pred_CNB"])[merged_labels_1["toxic"] == 0].mean()
print("Non-Toxic Accuracy Complement Naive Bayes model:", nontoxic_accuracy_CNB)

merged_labels_1.to_csv('final_dataframe.csv', index=False)






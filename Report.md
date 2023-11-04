## Observations

Note: The below observations are for a probability threshold of 0.5.

I used three models on the data. The following were the observations:

| Sl.no | Models                 | Accuracy on the entire dataset |
|-------|------------------------|---------------------------------|
| 1.    | Gaussian Naïve Bayes   | Kernel Died                     |
| 2.    | Multinomial Naïve Bayes| 91.75%                          |
| 3.    | Complement Naïve Bayes | 88.59%                          |

### Performance of Multinomial Naïve Bayes on Toxic and Non-Toxic Data

| Sl.no | Dataset           | Accuracy  |
|-------|-------------------|-----------|
| 1.    | Toxic Dataset     | 66.86%    |
| 2.    | Non-Toxic Dataset | 94.3%     |

### Performance of Complement Naïve Bayes on Toxic and Non-Toxic Data

| Sl.no | Dataset           | Accuracy  |
|-------|-------------------|-----------|
| 1.    | Toxic Dataset     | 76.15%    |
| 2.    | Non-Toxic Dataset | 89.89%    |

Performance of Multinomial Naïve Bayes when stop words are removed is 38%.

## Analysis

The following preprocessing was done on the data:

- Converted the text to lowercase to treat different case versions of the same word as the same word.
- Removed special characters from the dataset to remove noise and make it easier for the model to identify relevant patterns.

## Approach

After preprocessing the data, I tried to train and test the model on Gaussian Naïve Bayes. However, while making predictions on the test dataset, the kernel died every time, possibly due to the dataset's volume.

I used the Multinomial Naïve Bayes model, which is well-suited for data represented as word vector counts (CountVectorizer), and achieved an accuracy of 91.75% on the entire dataset. However, the accuracy of the model on the toxic dataset was 66.86%, while the accuracy on the non-toxic dataset was 94.3%.

Multinomial Naïve Bayes seemed to predict non-toxic dataset better than the toxic dataset. One possible reason for this could be the substantial class imbalance, where the ratio of toxic comments to non-toxic comments is approximately 1:10.

To address this, I used Complement Naïve Bayes, which is effective for imbalanced datasets. Although the overall accuracy dropped slightly from 91.75% to 88.59%, the accuracy for toxic data prediction increased significantly from 66.86% to 76.15%. Complement Naïve Bayes provided a more balanced prediction for the dataset.

## Adjusting the Decision Threshold

I experimented with different probability thresholds for predicting toxicity and observed the following:

| Sl.no | Model                 | Threshold | Dataset            | Accuracy |
|-------|-----------------------|-----------|--------------------|----------|
| 1.    | Multinomial Naïve Bayes | 0.5     | Complete Dataset   | 91.75    |
| 2.    | Complement Naïve Bayes  | 0.5     | Complete Dataset   | 88.59    |
| 3.    | Multinomial Naïve Bayes | 0.5     | Toxic Accuracy     | 66.86    |
| 4.    | Multinomial Naïve Bayes | 0.5     | Non-Toxic Accuracy | 94.37    |
| 5.    | Complement Naïve Bayes  | 0.5     | Toxic Accuracy     | 76.15    |
| 6.    | Complement Naïve Bayes  | 0.5     | Non-Toxic Accuracy | 89.89    |
| ...   | ...                   | ...       | ...                | ...      |

From the observations, it can be seen that the probability threshold of 0.5, 0.6, and 0.7 work well and give a balanced prediction for toxic and non-toxic datasets. The probability threshold of 0.5 was chosen as it provides the most balanced accuracy for toxic and non-toxic datasets.

Depending on the application and requirements, the thresholds can be adjusted to get the desired output.

## Additional Observations

In the final_dataframe.csv, there are comments labeled as non-toxic that seemed to be toxic. Our model performed well at predicting them as toxic. However, the test labels from Kaggle captured these comments as non-toxic. Some examples are listed below:

1. "I WILL BURN YOU TO HELL IF YOU REVOKE MY TALK PAGE ACCESS!!!!!!!!!!!!!" (toxic_pred_MNB: 1, toxic_pred_CNB: 1, toxic: 0)
2. "== Shameless Canvass == Hello, Diannaa! Thanks for blocking that horrible puke again. When I ""translate"" on your page, I get mad. He is just a total piece of garbage. Thanks again!" (toxic_pred_MNB: 1, toxic_pred_CNB: 1, toxic: 0)
3. "666 the devil will get you all !!!!!!!!!!!!!!!! !!!!!!!!!!!!!" (toxic_pred_MNB: 1, toxic_pred_CNB: 1, toxic: 0)

## Conclusion

This report presents the observations and analysis of the Naïve Bayes models applied to the toxic comment dataset. It explores the impact of different models and thresholds on toxicity predictions and offers insights into the challenges posed by imbalanced datasets.

Feel free to explore and modify the code to enhance your understanding and to improve the performance of the Naïve Bayes models.


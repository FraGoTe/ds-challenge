# Model Selection
 The classes are imbalance:
   - class 0 (18,294 samples)
   - class 1 (4,214 samples)
     
This imbalance can cause models to focus on correctly predicting class 0 but bad for class 1, leading to a poor recall and F1-scores for class 1.

| Model                                             | Precision (0) | Recall (0) | F1-score (0) | Precision (1) | Recall (1) | F1-score (1) | Accuracy | Macro Avg Precision | Macro Avg Recall | Macro Avg F1-score |
|---------------------------------------------------|---------------|------------|--------------|---------------|------------|--------------|----------|---------------------|------------------|--------------------|
| XGBoost                                           | 0.81          | 1.00       | 0.90         | 0.00          | 0.00       | 0.00         | 0.81     | 0.41                | 0.50             | 0.45               |
| Logistic Regression                               | 0.82          | 0.99       | 0.90         | 0.56          | 0.03       | 0.06         | 0.81     | 0.69                | 0.51             | 0.48               |
| XGBoost with Feature Importance & Balance         | 0.88          | 0.52       | 0.66         | 0.25          | 0.69       | 0.37         | 0.55     | 0.56                | 0.61             | 0.51               |
| XGBoost with Feature Importance (No Balance)      | 0.81          | 1.00       | 0.90         | 0.76          | 0.01       | 0.01         | 0.81     | 0.79                | 0.50             | 0.45               |
| Logistic Regression with Feature Importance & Balance | 0.88       | 0.52       | 0.65         | 0.25          | 0.69       | 0.36         | 0.55     | 0.56                | 0.60             | 0.51               |
| Logistic Regression with Feature Importance (No Balance) | 0.81       | 1.00       | 0.90         | 0.53          | 0.01       | 0.03         | 0.81     | 0.67                | 0.51             | 0.46               |


As we can see, our main objective is to predict delays. Even if the model accuracy is higher for some models like XGBoost, Logistic Regression, XGBoost with Feature Importance (No Balance), or Logistic Regression with Feature Importance (No Balance), recall and F1-score are too low then, in this case, I will choose *XGBoost with Feature Importance & Balance*




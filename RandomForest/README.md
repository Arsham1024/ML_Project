# Random Forest Implementation
#####
---
### Description:
#### Predicting the likelihood of heart attack with RF classifier.
---
### <a href='https://www.kaggle.com/fedesoriano/heart-failure-prediction'>Data Set</a>
###### 70-30 split was used for training and test sets.
---
### Approach:
###### The dataset used was already clean, no specific cleaning was done. However, lable encoding used for some features for a better presentation and results. 
###### Different parameters was used for RandomForestClassifier method: 
* RandomForestClassifier()
* RandomForestClassifier(max_depth = 5,10) 
* RandomForestClassifier(n_estimators = 10,20)
* RandomForestClassifier(max_depth, n_estimators)
---
### Plots:
* Confusion matrix: Showing the performance of the classifier (TN, TP, FN, FP)
---
### Accuracy:
###### Best accuracy 86.95% with n_estimators = 20
---
### Tools & Libraries:
* Python
* Numpy
* Pandas
* Sklearn
* Seoborn
* Matplotlib

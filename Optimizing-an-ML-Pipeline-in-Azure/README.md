# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
 This Project aims at making a binary prediction to find whether customers will join a Bank or Not.This dataset is related to direct marketing campaigns of a Portuguese banking sector. The campaigns were based on phone calls.

The best performing model was a ***Voting Ensemble*** of **Xgboost classifier** using standard scaler wrapper. This was found using Automl feature of Azure.

### Files Used to perform the Analysis are 

- Train.py
- Project.ipynb


We need to build a Machine learning model using skikit learn and tune the hyper parameters to find the best model using azure ML python SDK and Hyper Drive.
Post that we need to use the Azure AutoML Feature to find the best model and best Hyperparameters.

## Scikit-learn Pipeline
**The pipeline architecture**

- Initially we retrieve the dataset from the url provided using Azure TabularDatasetfactory class.
- Then we preprocess the dataset using the clean_data function in which some preprocessing steps were performed like converting categorical variable to binary encoding, one hot encoding,etc
- Then the dataset is split in ratio of 70:30 (train/test) for training and testing and sklearn's LogisticRegression Class is used to define Logistic Regression model.
- We then use inverse regularization(C) and maximum iterations(max_iter) hyperparamters which are tuned using Azure ML Hyper Drive to find the best combination for maximizing the accuracy.
- The classification algorithm used here is **Logistic Regression** with accuracy as the primary metric for classification which is completely defined in the train.py file
- Finally ,the best run of the hyperdrive is noted and the best model in the best run is saved.

**The benefits of the parameter sampler**

- Here I have used Random Parameter Sampling in the parameter sampler so that it can be used to provide random sampling over a hyperparameter search space.
- It also has the advantage of performing equally as Grid Search with lesser compute power requirements.

### Hyperparameters

- Inverse regularization parameter(C)- A control variable that retains strength modification of Regularization by being inversely positioned to the Lambda regulator. The relationship, would be that lowering C - would strengthen the Lambda regulator.

- No of iterations(max_iter):The number of times we want the learning to happen. This helps is solving high complex problems with large training hours

**The benefits of the early stopping policy** 
 
-  Early Stopping policy in HyperDriveConfig is useful in stopping the HyperDrive run if the accuracy of the model is not improving from the best accuracy by a certain defined amount after every given number of iterations
- Here we have used the BanditPolicy for early stopping policy with parameters slack_factor, slack_amount,Delay Evaluation and Evaluation Intervals, these are deined as:
   1. Slack_factor :- The ratio used to calculate the allowed distance from the best performing experiment run.
   2. Slack_amount :- The absolute distance allowed from the best performing run.
   3. evaluation_interval :- The frequency for applying the policy.
   4. delay_evaluation :- The number of intervals for which to delay the first policy evaluation. If specified, the policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation.

## AutoML
**AutoML means Automated Machine Learning**

1. AutmoML means that we can Automating the process, it reduces the time consumed by the training (Traditional) process. It also helps in performing iterative tasks of ML models.
2. With the help of AutoML we can accelerate the time taken for deployment of models into production with gread efficency.
3. The models used here were RandomForests,BoostedTrees,XGBoost,LightGBM,SGDClassifier,VotingEnsemble,SGD Classifier
4. It also used different pre- processing techniques like Standard Scaling, Min Max Scaling, Sparse Normalizer, MaxAbsScaler. It also has efficiently balanced the class Imabalance in the data.
5. It also has the feature of crossvalidation where number of cross_validation split is specified using which it performs validation on the dataset.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

- Overall,the difference in accuracy between the AutoML model and the Hyperdrive tuned custom model is not very much. AutoML accuracy was 0.91763 while the Hyperdrive accuracy was 0.91006

- With Respect to architecture AutoML was better than hyperdrive because it tried a lot of different models, which is quite impossible to do with Hyperdrive, as that would require us to create a new pipeline for each model.


## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**


### Reference:

Here are the references used for completing the project

- [Details on BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py)


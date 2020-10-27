# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run. The projects is performed on the [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and the Azure ML Studio.

## Summary
 This Project aims at making a binary prediction to find whether customers will join a Bank or Not.This dataset is related to direct marketing campaigns of a Portuguese banking sector. The campaigns were based on phone calls.

The best performing model was a **Voting Ensemble** of ***Xgboost classifier*** using standard scaler wrapper. This was found using Automl feature of Azure.

### Files Used to perform the Analysis are 

- Train.py
- Project.ipynb

We need to build a Machine learning model using skikit learn and tune the hyper parameters to find the best model using azure ML python SDK and Hyper Drive.
Post that we need to use the Azure AutoML Feature to find the best model and best Hyperparameters.At the end of this project, we want to create a solid understanding about the Azure ML Studio and some important standard practices of MLops inside Azure platform.

#### Steps followed for completion of Project

- Input the dataset into Registered Datasets;
- Create a compute instance;
- Using the Python SDK and a Scikit-learn model (Logistic Regression), it is developed an Azure ML pipeline;
- An AutoML run inside Azure ML Studio is then created to perform a series of experiments.

![Project Architecture](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Project%20Workflow.JPG)

The Architectural diagram of the Project is as given below :-

![Project Architecture](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Project%20Architecture.JPG)


## Scikit-learn Pipeline
**The pipeline architecture**

- Initially we retrieve the dataset from the url provided using Azure TabularDatasetfactory class.
- Then we preprocess the dataset using the clean_data function in which some preprocessing steps were performed like converting categorical variable to binary encoding, one hot encoding,etc
- Then the dataset is split in ratio of 70:30 (train/test) for training and testing and sklearn's LogisticRegression Class is used to define Logistic Regression model.
- We then use inverse regularization(C) and maximum iterations(max_iter) hyperparamters which are tuned using Azure ML Hyper Drive to find the best combination for maximizing the accuracy.
- The classification algorithm used here is **Logistic Regression** with accuracy as the primary metric for classification which is completely defined in the train.py file
- Finally ,the best run of the hyperdrive is noted and the best model in the best run is saved.

**The benefits of the parameter sampler**

- Here, I have used Random Parameter Sampling in the parameter sampler so that it can be used to provide random sampling over a hyperparameter search space.
- It also has the advantage of performing equally as Grid Search with lesser compute power requirements.

### Hyperparameters

- Inverse regularization parameter(C)- A control variable that retains strength modification of Regularization by being inversely positioned to the Lambda regulator. The relationship, would be that lowering C - would strengthen the Lambda regulator.

- No of iterations(max_iter):The number of times we want the learning to happen. This helps is solving high complex problems with large training hours.

- 

**The benefits of the early stopping policy** 
 
-  Early Stopping policy in HyperDriveConfig is useful in stopping the HyperDrive run if the accuracy of the model is not improving from the best accuracy by a certain defined amount after every given number of iterations
- Here we have used the BanditPolicy for early stopping policy with parameters slack_factor, slack_amount,Delay Evaluation and Evaluation Intervals, these are deined as:
   1. Slack_factor :- The ratio used to calculate the allowed distance from the best performing experiment run.
   2. Slack_amount :- The absolute distance allowed from the best performing run.
   3. evaluation_interval :- The frequency for applying the policy.
   4. delay_evaluation :- The number of intervals for which to delay the first policy evaluation. If specified, the policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation.
   
![Best metrices](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Hyperdrive%20ouputs/Best_metrics.JPG)

![Hyperdrive metrices](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Hyperdrive%20ouputs/Hyperdrive_metrics.JPG)

![Child run](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Hyperdrive%20ouputs/Child_runs.JPG)


## AutoML
**AutoML means Automated Machine Learning**

1. AutmoML means that we can Automating the process, it reduces the time consumed by the training (Traditional) process. It also helps in performing iterative tasks of ML models. It is known for its incredible flexibility
2. With the help of AutoML we can accelerate the time taken for deployment of models into production with great efficency.
3. The models used here were RandomForests,Boosted Trees,XGBoost,LightGBM,SGD Classifier,Voting Ensemble.
4. It also used different pre- processing techniques like Standard Scaling, Min Max Scaling, Sparse Normalizer, MaxAbsScaler. It also has efficiently balanced the class Imabalance in the data.
5. It also has the feature of crossvalidation where number of cross_validation split is specified using which it performs validation on the dataset.

In this project, we set the AutoML configuration with accuracy as primary metric and cross validation. The cross validation is important to avoid overfitting and helps generalize the model better. For computational reasons, in this experiment, the experiment timeout was set to 30 Minutes which have limited the number of Models that could be built.

![Automl models](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Automl%20outputs/Automl_models.JPG)

Best Auto ML model

![Best Automl model](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Automl%20outputs/Best_automl_model.JPG)

Best Metrices

![Best Metrices](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Automl%20outputs/Automl_bestmetrics.JPG)

Best Features

![Best Features](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Automl%20outputs/Automl_features.JPG)

![Best Features](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Automl%20outputs/Automl_features_boxplot.JPG)

### Findings from the AutoML project

- There are a lot of algorithms that has been tried and used by the automl run but the algorithms which got the highest accuracy was the VotingEnsemble, which is equal to approx. 91.72 percent. 
- There are different metrics that were ranging from 72 percent to 91 percent in which the lowest accuracy was from the algorithm called MinMaxScaler ExtremeRandomTrees. 
- We can also see that from the VotingEnsemble Model we have categorised the best Parameters that help the Bank retain it's clients/customers.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

- Overall,the difference in accuracy between the AutoML model and the Hyperdrive tuned custom model is very small. AutoML accuracy was 0.91763 while the Hyperdrive accuracy was 0.91006

- With Respect to architecture AutoML was better than hyperdrive because it tried a lot of different models, which is quite impossible to do with Hyperdrive, as that would require us to create a new pipeline for each model.

- The architecture of both pipelines are different, but the ideas are close: Load the data, instanciate the infrastructure to compute, set the parameters and call the compute method. The main difference is the that using AutoML we have infitine possibilities to increase the search for a better algorithm or a hyperparameter combination.


## Future work

We can notice that the dataset has Class Imbalance, we need to treat this by applying few balancing techniques like SMOTE,ADASYN and try to find further on this.
For improvements and future experiments it is important to explore other algorithms and validate more metrics. 
Accuracy in this case is very important, but using a **confusion matrix** can be important.We can also used balanced_accuracy which is a primary metric that calculates for the arithmetic mean of recall for each class. Another example is AUC_weighted, which gets the arithmetic mean of the score for each class , weighted by the true number of true instances in each class.
We can modify the experiments with other hyperparameters and increase the cross validation to enhance models performance and generalization.

Also, in this project, the experiment_timeout_minutes has been capped to only 30 minutes so there could only be a limited number of models that can be runned on that time period, so a longer timeout could have greater number of models to run and thus improving the performance.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**

![Deleting Compute](https://github.com/Aishwaryasasanapuri/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree-Program/blob/main/Optimizing-an-ML-Pipeline-in-Azure/Deleting_compute.JPG)

### Reference:

Here are the references used for completing the project

- [Details on BanditPolicy](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py)
- [More on Sklearn class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.sklearn.sklearn?view=azure-ml-py)
- [Argparse](https://realpython.com/command-line-interfaces-python-argparse/)
- [Details on hyperdrive config class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [Details on Run class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.Run?view=azure-ml-py)


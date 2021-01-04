*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.


# Operationalizing The Machine Learning Model In Azure

Here we use bank marketing [Dataset](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) and the Azure ML Studio..This dataset is related to direct marketing campaigns of a Portuguese banking sector. The campaigns were based on phone calls.The goal is to classify whether a client will subscribe to a term deposit or not.
The objective of this project is to build a machine learning model using Azure Container Services.We are provided with the banking dataset. The main steps of the project are as follows

1) Authentication
2) Automated ML Experiment
3) Deploy the best model
4) Enable logging
5) Swagger Documentation
6) Consume model endpoints
7) Create and publish a pipeline
9) Documentation

## Architectural Diagram
*TODO*: Provide an architectual diagram of the project and give an introduction of each step.

![Project Architecture](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Architecture%20overview.JPG)


## Key Steps
*TODO*: Write a short discription of the key steps. Remeber to include all the screencasts required to demonstrate key steps. 

## Authentication
Authentication is crucial for the continuous flow of operations. Continuous Integration and Delivery system (CI/CD) rely on uninterrupted flows. When authentication is not set properly, it requires human interaction and thus, the flow is interrupted. An ideal scenario is that the system doesn't stop waiting for a user to input a password. So whenever possible, it's good to use authentication with automation.
Authentication types 
1.Key Based 
2.Token based 
3.Interactive

## Register the dataset

We have to register the dataset using the local files or the URL provided 
- Navigate to the Datasets section in the Workspace and create a new dataset from webfile and submit the URL required for the dataset

![Dataset](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Dataset.JPG)

## Compute cluster
We have to build a compute cluster of type DS12_V2 for running the AutoML Run.
Maximum number of nodes are 5 and min number of nodes are 1.

![Compute](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Compute%20cluster.JPG)

## Auto ML

1. Create a new Auto ML experiment using the existing dataset with classification models and accuracy as parameter
2. The run undergoes around 73 different models and we get voting ensemble as best model with the base model XGBOOST with Maxabs scaling.

![Automlsetup](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Automl_setup.JPG)

Here, we enable explain best model option to obtain the best model

![Automlconfig](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/AutoML%20configuaration.JPG)

![Automl features](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/AutoML%20features.JPG)

![Automl setup](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Automl_setup.JPG)

## Best Model
After Automl run is complete we obtain the best model i.e. Voting Ensemble with accuracy of around 91%

![Bestmodel](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Bestmodel.JPG)

Here we can see the explaination of the dataset by best model

![explaination](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Bestmodelexplaination.JPG)

## Deploy the model

We can now deploy the best model. We can use azure Kubernetes service or azure container instane for the deployment.
Here we use Azure container Instance for deployment.We need to choose authentication method during the deployment method. 

![Deploy Model](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/deploymodel.JPG)

Once deployment is succeded an endpoint will be created with status showing as healthy in workspace.

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/deploystatus.JPG)

Healthy status of the model

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/deployedmodel.JPG)

## Aplication Insights

Enable Application Insights using the logs.py file and make sure that Endpoints section in Azure ML Studio, showing that “Application Insights enabled” says “true”.
We can see the status in application insights saying the failed requests, timed out requests etc.

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/logpyfile.JPG)

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/applicationinsights.JPG)

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Insights%20view.JPG)


## Consume Endpoints

- we use Swagger Documentation to get a simpilifed document to consume the HTTP API.
- Swagger runs on localhost showing the HTTP API methods and responses for the model.
- we get a Swagger JSon file from the endpoint which needs to be downloaded and placed in the folder containing swagger files serve.py and swagger.sh.
- After that we need to launch a local web server using serve script and lauch swagger using docker container by running swagger.sh

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Swagger_API.JPG)

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Parameters.JPG)

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Swagger_responses.JPG)

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/endpoint.JPG)

### Benchmarking API

Benchmarking HTTP APIs is used to find the average response time for a deployed model.A benchmark is used to create a baseline or acceptable performance measure.Here Apache Benchmark (ab) runs against the HTTP API using authentication keys to retrieve performance results

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/benchmark.sh)

## Create and Publish Endpoints

Pipeline form a basis of automation.Publishing a pipeline is the process of making a pipeline publicly available.
We can publish pipelines in Azure Machine Learning Studio and also with the Python SDK.
When a Pipeline is published, a public HTTP endpoint becomes available, allowing other services, including external ones, to interact with an Azure Pipeline.

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Pipeline_endpoints.JPG)

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Pipelinerunsummary.JPG)

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/Pipeline_endpoint_status.JPG)

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/rest%20endpoint.JPG)

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/published_pipeline.JPG)


*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

![](https://github.com/Aishwaryasasanapuri/sample/blob/main/Operationalizing%20Machine%20Learning/rundetails.JPG)

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

https://youtu.be/lUve_l99Tpg

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
1. Having more data on the experiment would have helpes us gain more insights
2. The exit criterion of the automl model is reduced to 1hour to save compute powe and resources which by increasing can lead to finding a better perofrming model
3. Increasing the number of cross validation can help in achieving better accuracy
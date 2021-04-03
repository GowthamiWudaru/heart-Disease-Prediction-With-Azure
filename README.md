# heart-Disease-Prediction-With-Azure

I predicted weather a person has a heart disease by using azure ML. I used automl and hyperdrive and deployed the model with best accuracy using Azure Container Instance(ACI)

## Dataset

### Overview
I got the data from <a href="https://archive.ics.uci.edu/ml/datasets/Heart+Disease">UCI Repository</a>. I used Cleveland database. It contains 14 attributes, namely, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num (the predicted attribute). 

### Task
The goal is to the find presence/absence of heart disease in the patient. The num attribute is integer valued from 0 (no presence) to 4. We concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).

### Access
I loaded the cleaveland data from <a href="https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/processed.cleveland.data"> my github repo</a> and cleaned it with processData.ipynb and loaded the data into <a href="https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/heartDisease.csv">heartDisease.csv</a> file and imported it in raw format in the code.

## Automated ML

I set experiment_timeout_minutes(time after which experiment is timed out), model_explainability(best model is explained), compute_cluster(multiple runs at a time) for automl run. The task is a classification(binary) task as we are trying to predict presence or absence of heart disease. I selected the primary metric as accuracy as the dataset is balanced.

![automlRunDetails](https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/images_for_readme/automlrunDetails.png)

### Results

The best model was VotingEnsemble with accuracy of 0.84870. Voting ensemble works by combining the predictions from multiple models. In classification, the final prediction is the majority vote of contributing models.The voting ensemble has parameters degree=3, gamma='scale', kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001.

![automlbestRun"](https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/images_for_readme/automlbestrun.png)

### Future Work

The model can be improved by further exploring the automl config(like adding custom FeaturizationConfig)

## Hyperparameter Tuning

I chose Logistic Regression model and tuned hyperparameters C(Inverse of regularization strength. Smaller values cause stronger regularization) and max-iter(Maximum number of iterations to converge). I used RandomParameterSampling with params max_iter(can have values 100,200,300,400) and C (can have 0.001, 0.01, 0.1, 1, 10, 100, 1000) and Bandit Policy with evaluation_interval(The frequency for applying the policy) as 2 and slack_factor(The ratio used to calculate the allowed distance from the best performing experiment run) as 0.1.

![hyperdriverunning](https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/images_for_readme/hyperdriverunning.png)

![hyperdriverunDetails](https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/images_for_readme/hyperDriveRunDetails.png)

### Results

The best model was a Logistic Regression model with an accuracy 0.88888888 for Regularization strength 100 amd max iteration 400

![hyperdrivebestRun](https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/images_for_readme/hyperdrivebestrun.png)

### Future Work

The model can be improved further by exploring different sampling techniques(grid sampling - grid sampling over a hyperparameter search space, bayesian sampling - tries to intelligently pick the next sample of hyperparameters, based on how the previous samples performed, such that the new sample improves the reported primary metric), early termination policy(Median stopping policy - based on running averages of the primary metric of all runs, Truncation selection policy - cancels a given percentage of runs at each evaluation interval)

## Model Deployment

The automl best run accuracy is 0.84870 and hyperdrive best run accuracy is 0.88888888. So, I deployed the LogisticRegression model using Azure container instance and loaded the script file and env file from automl run and changed the file path to point to LogisticRegression.pkl model

![EP healthy](https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/images_for_readme/EPhealthy.png)

We send the request to the EP by randomly getting 3 samples from the dataset, form a dictionary, converting it into json format and sending it to service by using service.run()

## ![Presentation](https://docs.google.com/presentation/d/1aPJaDqMD10CQkSxsWGI1c4ctl7mlrEiNVq3-N7LJ39U/edit?usp=sharing)

[![Youtube video](https://img.youtube.com/vi/NDxQLzwUdMg/0.jpg)](https://www.youtube.com/watch?v=NDxQLzwUdMg)

## Application Insights

Enabled Application Insights for the web service

![AI enabled](https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/images_for_readme/AIenabled.png)
![Application Dashboard](https://github.com/GowthamiWudaru/heart-Disease-Prediction-With-Azure/blob/main/images_for_readme/ApplicationInsights.png)

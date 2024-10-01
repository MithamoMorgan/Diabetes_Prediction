![](https://github.com/MithamoMorgan/Diabetes_Prediction/blob/master/Images/stop_diabetes.jpg)
Diabetes Prediction Project

## Table of Contents

1. [Overview](#Overview)</br>
2. [Tools](#Tools)</br>
3. [Dataset](#Dataset)</br>
4. [Problem Statement](#Problem-Statement)</br>
5. [Data Preprocessing](#Data-Preprocessing)</br>
6. [Modeling](#Modeling)</br>
7. [Evaluation Metrics](#Evaluation-Metrics)</br>
8. [Results](#Results)</br>
9. [Conclusion](#Conclusion)</br>
10. [Future Work](#Future-Work)</br>
11. [How to Use](#How-to-use)</br>
12. [Requirements](#Requirements)</br>
13. [App Overview](#App-Overview)

## Overview

The project is aimed to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans.

## Tools:
* **Pandas:** Data manipulation and analysis.
* **Numpy:** For numerical computation.
* **Matplotlib & seaborn:** Visualization
* **Scikit-Learn:** Implementing various machine learning algorithms.
* **XGBoost:** For gradient boosting.
* **Streamlit:** Model deployment.
* **Jupyter Notebook:** Documenting process results.
* **VSCode:** Development environment.
## Dataset
I downloaded the data used in this project from kaggle. Here is the [link](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

The data used include features such as: 
* age - important factor as diabetes is more commonly diagnosed in older adults.
* gender - Can have an impact on individual's susceptibility to diabetes.
* body mass index(BMI) - Measure of body fat based on weight and height.
* hypertension - Medical condition in which blood preassure in the alteries is persistently elevated. 
* heart disease - Associated with an increased risk of developing diabetes.
* smoking history - Smoking history is considered a risk factor for diabetes.
* HbA1c - Measure of person's average blood sugar level over past 2-3 months.
* glucose level - Amount of glucose in the bloodstream at a given time.

## Problem Statement

The objective is to classify individuals as diabetic or non-diabetic using machine learning algorithms based on their medical history and demographic information.
This is a Binary Classification problem.

## Data Preprocessing

**Handling Duplicates:**
The dataset had 3,854 duplicates, which were handled by dropping.

```python
df.drop_duplicates()
```

**Feature Scaling:**

I applied the `StandardScaler` to standardize the continuous features, ensuring a mean of 0 and a standard deviation of 1. First, I fitted the scaler on the training set and transformed it. Then, I used the same scaling on the testing set to maintain consistency, which helps improve the model's performance and robustness.
```python
    # initialize the StandardScaler
    scaler = StandardScaler()

    # Fitting the scaler on the training data
    X_train = scaler.fit_transform(X_train)

    # Apply the transformations from training set on the testing set
    X_test = scaler.transform(X_test)
```
**Encoding:** 

I utilized the `LabelEncoder` from the `sklearn.preprocessing` module to convert the categorical variables into a format suitable for machine learning algorithms.
```python
# initialize LabelEncoder instance
label_encoder = LabelEncoder()

# transofrm each cateorical variable
categorical_variables = df.select_dtypes(include = 'O')
for column in categorical_variables:
    df[column] = label_encoder.fit_transform(df[column])
```

**Train-Test Split:**

I split the dataset into training and testing sets using the `train_test_split` function from `sklearn`. I allocated 70% of the data for training and 30% for testing, while setting a random state for reproducibility. This approach allows us to train the model on one subset and evaluate its performance on an unseen subset, enhancing the model's reliability.
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
````

## Modeling

**Models Tested:**

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. XGBoost Classifier

## Evaluation Metrics

The metrics used to evaluate the model performance were:

1. Accuracy
2. Precision
3. Recall (focused on recall since the priority was detecting positive cases)
4. F1-Score
5. ROC-AUC

## Results

* For accuracy, the Gradient Boosting Classifier is the best model.
* For recall, the Decision Tree performs the best.
* For precision, the Gradient Boosting Classifier also excels with the highest score.
* For overall performance measured by the F1 score, the XGBoost Classifier is the top performer.

| Name                       | Accuracy Score | Precision Score | Recall Score | F1 Score  |
|----------------------------|----------------|-----------------|--------------|-----------|
| Logistic Regression        | 0.958778       | 0.866808        | 0.636399     | 0.733945  |
| Decision Tree              | 0.950250       | 0.712110        | 0.743888     | 0.727652  |
| Random Forest              | 0.969248       | 0.947564        | 0.694218     | 0.801344  |
| Gradient Boosting Classifier| 0.971398      | 0.986667        | 0.689173     | 0.811515  |
| XGBoost Classifier         | 0.970982       | 0.962274        | 0.702755     | 0.812290  |

## Conclusion

The Decision Tree is the best choice for predicting diabetes since recall is the priority, as it achieves the highest recall score. However, if a balance of performance metrics is needed, the XGBoost Classifier remains a strong contender.

## Future Work

potential improvements or extensions of the project:

**More Data:** Collecting additional data to improve model generalization.

**Model Improvements:** Trying advanced techniques like Neural Networks or AutoML for further improvement.

## How to Use

You can run this project using these easy steps.

1. Clone the repository using:

   `git clone https://github.com/MithamoMorgan/Diabetes_Prediction.git`


2. Install the necessary dependencies (listed in th requirements.txt fil).


3. Run the Jupyter Notebook or Streamlit app :
   
   * To launch the Streamlit app, use the following command in your terminal:

      `streamlit run app.py`

   * Alternatively, you can run your Jupyter Notebook directly in Visual Studio Code, which supports interactive notebook execution.
  
## Requirements
* pandas</br>
* numpy</br>
* matplotlib</br>
* seaborn</br>
* scikit-learn</br>
* xgboost</br>
* streamlit</br>
* jupyter

Refer the requirements text file for automated Installation: It allows users to install all necessary packages quickly using a single command (`pip install -r requirements.txt`), making setup much easier.

## App Overview
This app can assist medical professionals in making a diagnosis but should not be used as a substitute for a professional diagnosis.</br>
*i:*
![](https://github.com/MithamoMorgan/Diabetes_Prediction/blob/master/Images/diabets_app.png)
*ii*
![](https://github.com/MithamoMorgan/Diabetes_Prediction/blob/master/Images/non_diabetic.jpg)


Work in Progress👷‍♂️⚙️🚧...


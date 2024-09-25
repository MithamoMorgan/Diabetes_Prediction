# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib

# import the dataset
df = pd.read_csv('clean_diabetes_data')

# Create LabelEncoder instance
label_encoder = LabelEncoder()

# Label encode categorical variables
categorical_variables = df.select_dtypes(include = ['O'])

for column in categorical_variables:
    df[column] = label_encoder.fit_transform(df[column])

# Define features and target variables
X = df.drop('diabetes', axis = 1)
y = df['diabetes']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)

# initialize the standardscaler
scaler = StandardScaler()

# Apply the transformation to the training and test data
X = scaler.fit_transform(X)

# create model intance
xgb_model = XGBClassifier(random_state =42)

# fit the data
xgb_model.fit(X_train, y_train)

# predict
xgb_y_pred = xgb_model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, xgb_y_pred)
class_report = classification_report(y_test, xgb_y_pred)

# Print the metrics
print("Confusion matrix:\n", conf_matrix)
print("Classification report:\n", class_report)
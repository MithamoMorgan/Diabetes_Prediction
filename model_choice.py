# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib

# import the dataset
df = pd.read_csv('clean_diabetes_data.csv')

#Select categorical columns
categorical_columns = df.select_dtypes(include = 'O').columns

# Initialize LabelEncoder instance
label_encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Define features and target variables
X = df.drop('diabetes', axis = 1)
y = df['diabetes']

# Split the data into training and testing tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# initialize the StandardScaler
scaler = StandardScaler()

# Fitting the scaler on the training data
X_train = scaler.fit_transform(X_train)

# Apply the transformations from training set on the testing set
X_test = scaler.transform(X_test)

# Create a random forest model with class weights balanced
random_forest = RandomForestClassifier(class_weight = 'balanced', random_state = 42)

# Fit the model to the training data
random_forest.fit(X_train, y_train)

# Make predictions
rf_y_pred = random_forest.predict(X_test)

# Evaluate the model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
rf_classification_rep = classification_report(y_test, rf_y_pred)

# Printing the results
print("Accuracy:", rf_accuracy)
print("Confusion Matrix:\n", rf_conf_matrix)
print("Classification Report:\n", rf_classification_rep)

# Save the model
joblib.dump(random_forest, 'random_forest_model.pkl')

# Save the scaler
scaler = joblib.dump(scaler, 'scaler.pkl')

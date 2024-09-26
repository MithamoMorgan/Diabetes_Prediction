# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # USe transorm() on X_test to avoid data leakage

# create model intances

models = {'LogisticRegression': LogisticRegression(random_state= 42),
          'DecisionTreeClassifier': DecisionTreeClassifier(random_state= 42),
          'RandomForestClassifier': RandomForestClassifier(random_state= 42),
          'GradientBoostingClassifier': GradientBoostingClassifier(random_state= 42),
          'XGBClassifier': XGBClassifier()}

# Create an empty list to store metrics and empty dict to store predictions
metrics = []
predictions = {}

# Loop through the models, fit them and store metrics in the DataFrame
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Store predictions for later use (in confusion matrix loop)
    predictions[name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Append the metrics to the metrics_df
    metrics.append({'Model': name,
                       'Accuracy': accuracy,
                       'Precision': precision,
                       'Recall': recall,
                       'F1-Score': f1})
    
# Convert the list to a DataFrame
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Create confusion matrix for each model
for name, y_pred in predictions.items():  # Ensures consistency(using same predictions for metrics and confusion matrix)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(name+"'s confusion Matrix:\n", conf_matrix)

# Save the model
print("saving the model...")
joblib.dump(XGBClassifier, 'gbc.pkl')
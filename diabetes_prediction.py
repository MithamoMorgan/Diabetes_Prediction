# import necesseeary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score,f1_score,
                             confusion_matrix, classification_report)
from xgboost import XGBClassifier
import joblib


def encode_categorical_features(df):
    """Encode categorical features."""

    label_encoder = LabelEncoder()
    categorical_variables = df.select_dtypes(include = 'O')
    for column in categorical_variables:
        df[column] = label_encoder.fit_transform(df[column])

    return df

def define_features_target_variable(df):
    """Define features and target variable."""

    X = df.drop('diabetes', axis = 1)
    y = df['diabetes']

    return X, y

def split_data(X, y):
    """Split the data."""

    return train_test_split(X, y, test_size = 0.3, random_state = 42)

def scale_data(X_test, X_train):
    """Fit the scaler on the training data and apply the transformations from training
    set on the testing set."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def initialize_models():
    """Create model instances"""
    
    models = {'LogisticRegression': LogisticRegression(random_state = 42),
         'DecisionTree': DecisionTreeClassifier(random_state = 42),
         'RandomForest' : RandomForestClassifier(random_state = 42),
         'GradientBoostingClassifier' : GradientBoostingClassifier(random_state = 42),
         'XGBClassifier' : XGBClassifier(random_state = 42)}
    
    return models
    
def train_and_predict(models, X_train, X_test, y_train):
    """Loop through the models, train, and predict"""

    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

    return predictions

def evaluate_the_models(models, predictions, y_test):
    """Evaluate the model metrics, print the confusion matrix,
    and save the model with the highest recall"""

    metrics = []
    highest_recall = 0
    best_model = None

    for name, y_pred in predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f"{name}:\n{conf_matrix}\n")

        metrics.append({
                        "Name" : name,
                        "Accuracy_score" : accuracy,
                        "Precision_score" : precision,
                        "recall_score" : recall,
                        "F1_score" : f1})
        
        # Track the best model based on recall
        if recall > highest_recall:
            highest_recall = recall
            best_model = models[name]

    metrics_df = pd.DataFrame(metrics)

    return best_model, metrics_df

def main():
    # Load the datase
    df = pd.read_csv('clean_diabetes_data.csv')

    # Encode categorical features
    encoded_df = encode_categorical_features(df)

    # Define features and target variable
    X, y = define_features_target_variable(df)

    # Split the data
    X_test, X_train, y_test, y_train = split_data(X, y)

    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_test, X_train)

    # Create model instance
    models = initialize_models()

    # Train and predict
    predictions = train_and_predict(models, X_train_scaled, X_test_scaled, y_train)

    # Evaluate the model
    best_model, metrics_df = evaluate_the_models(models, predictions, y_test)

    # Save the best model and scaler
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print("Metrics DataFrame:\n")
    print(metrics_df)


# Call the main function
if __name__ == '__main__':
    main()
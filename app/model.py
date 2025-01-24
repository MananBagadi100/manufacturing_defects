import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Define model training and prediction logic
class DowntimeModel:
    def __init__(self):
        self.model = RandomForestClassifier(class_weight="balanced", random_state=42)

    def train(self, dataset_path):
        # Load and preprocess the dataset
        dataset = pd.read_csv(dataset_path)
        dataset['Failure'] = dataset['Failure Type'].apply(lambda x: 0 if x == 'No Failure' else 1)
        dataset = dataset.drop(['UDI', 'Product ID', 'Failure Type'], axis=1)
        
        # Encode categorical column
        le = LabelEncoder()
        dataset['Type'] = le.fit_transform(dataset['Type'])
        
        # Features and target
        X = dataset.drop('Failure', axis=1)
        y = dataset['Failure']

        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, 'downtime_model.pkl')
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report

    def predict(self, input_data):
        # Predict using the trained model
        model = joblib.load('downtime_model.pkl')
        prediction = model.predict([input_data])
        confidence = max(model.predict_proba([input_data])[0])
        return {"Downtime": "Yes" if prediction[0] == 1 else "No", "Confidence": confidence}

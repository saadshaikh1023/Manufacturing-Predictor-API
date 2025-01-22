import xgboost as xgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib
import config

class ModelTrainer:
    def __init__(self):
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('scaler', StandardScaler()),
            ('classifier', xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42
            ))
        ])
        self.base_features = ["Temperature", "Run_Time"]
        
    def preprocess_data(self, df):
        # Add time-based features
        df['Time_Until_Maintenance'] = 300 - df['Run_Time']  # Assuming 300 is max runtime
        df['Temperature_Rate'] = df['Temperature'] / df['Run_Time']
        df['Critical_Zone'] = ((df['Temperature'] > 85) & (df['Run_Time'] > 200)).astype(int)
        
        features = self.base_features + ['Time_Until_Maintenance', 'Temperature_Rate', 'Critical_Zone']
        return df, features

    def train(self):
        df = pd.read_csv(f"{config.UPLOAD_FOLDER}/sample_data.csv")
        df, self.features = self.preprocess_data(df)
        
        X = df[self.features]
        y = df['Downtime_Flag']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Hyperparameter tuning
        param_grid = {
            'classifier__max_depth': [4, 5, 6],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__n_estimators': [200, 300],
            'classifier__min_child_weight': [1, 3],
            'classifier__subsample': [0.8, 0.9]
        }
        
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 2),
            "f1_score": round(f1_score(y_test, y_pred, average='weighted'), 2)
        }
        
        joblib.dump(self.model, config.MODEL_PATH)
        return metrics
    
    def predict(self, data):
        data['Time_Until_Maintenance'] = 300 - data['Run_Time']
        data['Temperature_Rate'] = data['Temperature'] / data['Run_Time']
        data['Critical_Zone'] = 1 if (data['Temperature'] > 85 and data['Run_Time'] > 200) else 0
        
        model = joblib.load(config.MODEL_PATH)
        features = pd.DataFrame([data])[self.features]
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0])
        
        return {
            "Downtime": "Yes" if prediction == 1 else "No",
            "Confidence": float(confidence)
        }
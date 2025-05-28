#!/usr/bin/env python3
# handwritten_cross_subject_validation_fixed.py - Fixed version with numpy issues resolved

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import warnings
warnings.filterwarnings('ignore')

def ensure_clean_arrays(X, y):
    """Ensure arrays are clean and properly typed"""
    # Convert to proper types
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int32)
    
    # Handle NaN and Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Ensure finite values
    X = np.clip(X, -1e10, 1e10)
    
    return X, y

def train_models_fixed(X_train, y_train):
    """Train models with proper error handling"""
    print("  🔧 Training models (fixed version)...")
    
    # Clean data
    X_train, y_train = ensure_clean_arrays(X_train, y_train)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Define models with conservative parameters
    models = {
        'svm': SVC(probability=True, random_state=42, C=1.0, gamma='scale'),
        'lr': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
        'rf': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5, 
                                   min_samples_split=5, min_samples_leaf=2)
    }
    
    trained_models = {}
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            trained_models[name] = model
            print(f"    ✅ {name.upper()} trained successfully")
        except Exception as e:
            print(f"    ❌ {name.upper()} training failed: {str(e)}")
    
    # Create ensemble if we have multiple models
    if len(trained_models) > 1:
        try:
            voting_clf = VotingClassifier(
                estimators=[(name, model) for name, model in trained_models.items()],
                voting='soft'
            )
            voting_clf.fit(X_train_scaled, y_train)
            trained_models['ensemble'] = voting_clf
            print(f"    ✅ ENSEMBLE trained successfully")
        except Exception as e:
            print(f"    ❌ ENSEMBLE training failed: {str(e)}")
    
    return trained_models, scaler

def evaluate_models_fixed(models, scaler, X_test, y_test):
    """Evaluate models with proper error handling"""
    # Clean data
    X_test, y_test = ensure_clean_arrays(X_test, y_test)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test_scaled)
            
            # Ensure predictions are proper integers
            y_pred = np.array(y_pred, dtype=np.int32)
            
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test_scaled)
                y_proba = np.array(y_proba, dtype=np.float64)
            else:
                y_proba = None
            
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'accuracy': float(accuracy),
                'predictions': y_pred,
                'probabilities': y_proba,
                'confusion_matrix': cm
            }
            print(f"    ✅ {name.upper()} evaluated: {accuracy:.4f}")
            
        except Exception as e:
            print(f"    ❌ {name.upper()} evaluation failed: {str(e)}")
            # Create dummy result
            results[name] = {
                'accuracy': 0.0,
                'predictions': np.zeros_like(y_test),
                'probabilities': None,
                'confusion_matrix': np.zeros((2, 2))
            }
    
    return results

print("✅ Fixed validator functions created!")

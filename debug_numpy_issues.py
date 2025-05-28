#!/usr/bin/env python3
# debug_numpy_issues.py - Debug numpy array conversion issues

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

def check_environment():
    """Check environment and package versions"""
    print("🔍 Environment Check")
    print("=" * 40)
    
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    
    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn not available")
    
    try:
        import pandas as pd
        print(f"Pandas version: {pd.__version__}")
    except ImportError:
        print("❌ Pandas not available")

def debug_array_issues():
    """Debug specific array conversion issues"""
    print(f"\n🔧 Debugging Array Issues")
    print("=" * 40)
    
    # Load the problematic data
    try:
        eeg_data = np.load('handwritten_eeg_data.npy')
        labels = np.load('handwritten_session_labels.npy')
        
        print(f"✅ Data loaded successfully")
        print(f"EEG data type: {eeg_data.dtype}")
        print(f"EEG data shape: {eeg_data.shape}")
        print(f"Labels data type: {labels.dtype}")
        print(f"Labels shape: {labels.shape}")
        
        # Check for problematic values
        print(f"\nData Quality Check:")
        print(f"EEG - Has NaN: {np.isnan(eeg_data).any()}")
        print(f"EEG - Has Inf: {np.isinf(eeg_data).any()}")
        print(f"EEG - Min: {np.min(eeg_data)}")
        print(f"EEG - Max: {np.max(eeg_data)}")
        
        print(f"Labels - Has NaN: {np.isnan(labels).any()}")
        print(f"Labels - Unique values: {np.unique(labels)}")
        
        return eeg_data, labels
        
    except FileNotFoundError:
        print("❌ Handwritten data not found")
        return None, None

def test_simple_rf():
    """Test Random Forest with simple synthetic data"""
    print(f"\n🧪 Testing Random Forest with Synthetic Data")
    print("=" * 50)
    
    # Create simple synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float64)
    y = np.random.randint(0, 2, 100).astype(np.int32)
    
    print(f"Synthetic data:")
    print(f"X shape: {X.shape}, dtype: {X.dtype}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}")
    
    try:
        # Test Random Forest
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        
        predictions = rf.predict(X)
        print(f"✅ Random Forest works with synthetic data")
        print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Random Forest failed with synthetic data: {str(e)}")
        return False

def test_rf_with_real_data(eeg_data, labels):
    """Test Random Forest with real handwritten data"""
    print(f"\n🧪 Testing Random Forest with Real Data")
    print("=" * 50)
    
    if eeg_data is None or labels is None:
        print("❌ No real data available")
        return False
    
    # Take a small subset for testing
    subset_size = 1000
    indices = np.random.choice(len(eeg_data), min(subset_size, len(eeg_data)), replace=False)
    
    X_subset = eeg_data[indices]
    y_subset = labels[indices]
    
    print(f"Subset data:")
    print(f"X shape: {X_subset.shape}, dtype: {X_subset.dtype}")
    print(f"y shape: {y_subset.shape}, dtype: {y_subset.dtype}")
    
    # Convert to ensure proper types
    X_clean = np.array(X_subset, dtype=np.float64)
    y_clean = np.array(y_subset, dtype=np.int32)
    
    print(f"Cleaned data:")
    print(f"X shape: {X_clean.shape}, dtype: {X_clean.dtype}")
    print(f"y shape: {y_clean.shape}, dtype: {y_clean.dtype}")
    
    # Check for problematic values
    if np.isnan(X_clean).any():
        print("⚠️ Found NaN values, replacing with 0")
        X_clean = np.nan_to_num(X_clean)
    
    if np.isinf(X_clean).any():
        print("⚠️ Found Inf values, replacing with finite values")
        X_clean = np.nan_to_num(X_clean)
    
    try:
        # Test Random Forest
        rf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
        rf.fit(X_clean, y_clean)
        
        predictions = rf.predict(X_clean)
        probabilities = rf.predict_proba(X_clean)
        
        print(f"✅ Random Forest works with real data")
        print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
        print(f"Probabilities shape: {probabilities.shape}, dtype: {probabilities.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Random Forest failed with real data: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test the feature extraction process"""
    print(f"\n🧪 Testing Feature Extraction Process")
    print("=" * 50)
    
    try:
        eeg_data = np.load('handwritten_eeg_data.npy')
        
        # Take a small window for testing
        window_size = 500
        window_data = eeg_data[:window_size]
        
        print(f"Window data shape: {window_data.shape}")
        print(f"Window data dtype: {window_data.dtype}")
        
        # Extract features like in the main script
        features = []
        
        # Statistical features
        features.extend(np.mean(window_data, axis=0))
        features.extend(np.std(window_data, axis=0))
        features.extend(np.min(window_data, axis=0))
        features.extend(np.max(window_data, axis=0))
        
        # Frequency domain features
        fft_data = np.abs(np.fft.fft(window_data, axis=0))
        freq_bands = [(0, 4), (4, 8), (8, 13), (13, 30), (30, 50)]
        
        for low, high in freq_bands:
            band_power = np.mean(fft_data[low:high], axis=0)
            features.extend(band_power)
        
        features_array = np.array(features, dtype=np.float64)
        
        print(f"✅ Feature extraction successful")
        print(f"Features shape: {features_array.shape}")
        print(f"Features dtype: {features_array.dtype}")
        print(f"Features has NaN: {np.isnan(features_array).any()}")
        print(f"Features has Inf: {np.isinf(features_array).any()}")
        
        # Clean features
        if np.isnan(features_array).any() or np.isinf(features_array).any():
            features_array = np.nan_to_num(features_array)
            print(f"🔧 Cleaned features")
        
        return features_array
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_complete_pipeline():
    """Test the complete pipeline with fixes"""
    print(f"\n🧪 Testing Complete Pipeline with Fixes")
    print("=" * 60)
    
    try:
        # Load data
        eeg_data = np.load('handwritten_eeg_data.npy')
        session_labels = np.load('handwritten_session_labels.npy')
        
        # Create a simple test case
        # Take data from session 0 and 1
        session_0_mask = session_labels == 0
        session_1_mask = session_labels == 1
        
        # Take subset from each session
        n_samples = 500
        session_0_data = eeg_data[session_0_mask][:n_samples]
        session_1_data = eeg_data[session_1_mask][:n_samples]
        
        # Create features for each sample
        def extract_simple_features(data_window):
            """Extract simple statistical features"""
            features = []
            features.extend(np.mean(data_window, axis=0))  # 64 features
            features.extend(np.std(data_window, axis=0))   # 64 features
            return np.array(features, dtype=np.float64)
        
        # Extract features for windows
        window_size = 100
        step_size = 50
        
        all_features = []
        all_labels = []
        
        # Process session 0 (label 0)
        for i in range(0, len(session_0_data) - window_size, step_size):
            window = session_0_data[i:i+window_size]
            features = extract_simple_features(window)
            all_features.append(features)
            all_labels.append(0)
        
        # Process session 1 (label 1)
        for i in range(0, len(session_1_data) - window_size, step_size):
            window = session_1_data[i:i+window_size]
            features = extract_simple_features(window)
            all_features.append(features)
            all_labels.append(1)
        
        # Convert to arrays
        X = np.array(all_features, dtype=np.float64)
        y = np.array(all_labels, dtype=np.int32)
        
        print(f"Pipeline data:")
        print(f"X shape: {X.shape}, dtype: {X.dtype}")
        print(f"y shape: {y.shape}, dtype: {y.dtype}")
        print(f"Label distribution: {np.bincount(y)}")
        
        # Clean data
        X = np.nan_to_num(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training data: {X_train_scaled.shape}")
        print(f"Test data: {X_test_scaled.shape}")
        
        # Test models
        models = {
            'rf': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5),
        }
        
        results = {}
        for name, model in models.items():
            try:
                print(f"\n🔧 Testing {name.upper()}...")
                
                # Fit model
                model.fit(X_train_scaled, y_train)
                print(f"  ✅ Training successful")
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
                
                accuracy = np.mean(y_pred == y_test)
                
                print(f"  ✅ Prediction successful")
                print(f"  📊 Accuracy: {accuracy:.4f}")
                print(f"  📊 Predictions dtype: {y_pred.dtype}")
                print(f"  📊 Probabilities dtype: {y_proba.dtype}")
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_proba
                }
                
            except Exception as e:
                print(f"  ❌ {name.upper()} failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if results:
            print(f"\n✅ Complete pipeline test successful!")
            return True
        else:
            print(f"\n❌ Complete pipeline test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_fixed_validator():
    """Create a fixed version of the cross-subject validator"""
    print(f"\n🔧 Creating Fixed Cross-Subject Validator")
    print("=" * 50)
    
    fixed_code = '''#!/usr/bin/env python3
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
'''
    
    with open('handwritten_cross_subject_validation_fixed.py', 'w') as f:
        f.write(fixed_code)
    
    print("✅ Fixed validator saved as 'handwritten_cross_subject_validation_fixed.py'")

def main():
    """Main debugging function"""
    print("🔧 NumPy Array Conversion Issues Debugging")
    print("=" * 60)
    
    # Step 1: Check environment
    check_environment()
    
    # Step 2: Debug array issues
    eeg_data, labels = debug_array_issues()
    
    # Step 3: Test simple RF
    simple_rf_works = test_simple_rf()
    
    # Step 4: Test RF with real data
    if eeg_data is not None:
        real_rf_works = test_rf_with_real_data(eeg_data, labels)
    else:
        real_rf_works = False
    
    # Step 5: Test feature extraction
    features = test_feature_extraction()
    
    # Step 6: Test complete pipeline
    pipeline_works = test_complete_pipeline()
    
    # Step 7: Create fixed validator
    create_fixed_validator()
    
    # Summary
    print(f"\n📊 Debugging Summary")
    print("=" * 30)
    print(f"Simple RF works: {'✅' if simple_rf_works else '❌'}")
    print(f"Real data RF works: {'✅' if real_rf_works else '❌'}")
    print(f"Feature extraction works: {'✅' if features is not None else '❌'}")
    print(f"Complete pipeline works: {'✅' if pipeline_works else '❌'}")
    
    if pipeline_works:
        print(f"\n✅ Environment is working correctly!")
        print(f"🔧 The issue was likely in data handling/type conversion")
        print(f"📝 Use the fixed validator for better results")
    else:
        print(f"\n❌ Environment has issues that need addressing")
        print(f"🔧 Consider updating packages or checking data quality")

if __name__ == "__main__":
    main()

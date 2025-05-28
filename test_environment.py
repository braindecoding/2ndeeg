#!/usr/bin/env python3
# test_environment.py - Test if environment is working

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys

def test_basic_functionality():
    """Test basic ML functionality"""
    print("🧪 Testing Basic ML Functionality")
    print("=" * 40)
    
    try:
        # Check versions
        import sklearn
        print(f"✅ Scikit-learn: {sklearn.__version__}")
        print(f"✅ NumPy: {np.__version__}")
        
        # Create test data
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float64)
        y = np.random.randint(0, 2, 100).astype(np.int32)
        
        print(f"✅ Test data created: X{X.shape}, y{y.shape}")
        
        # Test models
        models = {
            'LogisticRegression': LogisticRegression(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        for name, model in models.items():
            try:
                model.fit(X, y)
                pred = model.predict(X)
                acc = accuracy_score(y, pred)
                print(f"✅ {name}: {acc:.3f}")
            except Exception as e:
                print(f"❌ {name}: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {str(e)}")
        return False

def test_handwritten_data():
    """Test with handwritten data"""
    print(f"\n🧪 Testing with Handwritten Data")
    print("=" * 40)
    
    try:
        # Load data
        eeg_data = np.load('handwritten_eeg_data.npy')
        labels = np.load('handwritten_session_labels.npy')
        
        print(f"✅ Data loaded: EEG{eeg_data.shape}, Labels{labels.shape}")
        
        # Take small subset
        subset_size = 1000
        indices = np.random.choice(len(eeg_data), subset_size, replace=False)
        X_subset = eeg_data[indices].astype(np.float64)
        y_subset = labels[indices].astype(np.int32)
        
        # Create binary classification (session 0 vs others)
        y_binary = (y_subset > 0).astype(np.int32)
        
        print(f"✅ Subset created: X{X_subset.shape}, y{y_binary.shape}")
        print(f"✅ Label distribution: {np.bincount(y_binary)}")
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        
        # Test Random Forest specifically
        rf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
        rf.fit(X_scaled, y_binary)
        pred = rf.predict(X_scaled)
        acc = accuracy_score(y_binary, pred)
        
        print(f"✅ RandomForest with real data: {acc:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Handwritten data test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔧 Environment Testing")
    print("=" * 30)
    
    basic_ok = test_basic_functionality()
    handwritten_ok = test_handwritten_data()
    
    print(f"\n📊 Test Results:")
    print(f"Basic functionality: {'✅' if basic_ok else '❌'}")
    print(f"Handwritten data: {'✅' if handwritten_ok else '❌'}")
    
    if basic_ok and handwritten_ok:
        print(f"\n🎉 Environment is working correctly!")
        print(f"✅ Ready to run cross-subject validation")
    else:
        print(f"\n❌ Environment has issues")

if __name__ == "__main__":
    main()

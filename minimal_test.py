#!/usr/bin/env python3
# minimal_test.py - Minimal test to check what's working

import numpy as np
import sys
import os

def check_files():
    """Check if required files exist"""
    print("📂 Checking files...")
    
    files_to_check = [
        'handwritten_eeg_data.npy',
        'handwritten_session_labels.npy'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024**2)  # MB
            print(f"✅ {file}: {size:.1f} MB")
        else:
            print(f"❌ {file}: Not found")
    
    return all(os.path.exists(f) for f in files_to_check)

def test_data_loading():
    """Test data loading"""
    print("\n📊 Testing data loading...")
    
    try:
        print("Loading EEG data...")
        eeg_data = np.load('handwritten_eeg_data.npy')
        print(f"✅ EEG data: {eeg_data.shape}, {eeg_data.dtype}")
        
        print("Loading labels...")
        labels = np.load('handwritten_session_labels.npy')
        print(f"✅ Labels: {labels.shape}, {labels.dtype}")
        
        print(f"✅ Data loaded successfully!")
        return eeg_data, labels
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return None, None

def test_sklearn():
    """Test sklearn functionality"""
    print("\n🧪 Testing sklearn...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        print("✅ Sklearn imports successful")
        
        # Test with simple data
        X = np.random.randn(50, 10).astype(np.float64)
        y = np.random.randint(0, 2, 50).astype(np.int32)
        
        # Test LogisticRegression
        lr = LogisticRegression(random_state=42)
        lr.fit(X, y)
        pred_lr = lr.predict(X)
        print(f"✅ LogisticRegression: {len(pred_lr)} predictions")
        
        # Test RandomForest
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X, y)
        pred_rf = rf.predict(X)
        print(f"✅ RandomForest: {len(pred_rf)} predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ Sklearn test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_data(eeg_data, labels):
    """Test with small subset of real data"""
    print("\n🧪 Testing with real data subset...")
    
    if eeg_data is None or labels is None:
        print("❌ No data available")
        return False
    
    try:
        # Take very small subset
        subset_size = 100
        indices = np.random.choice(len(eeg_data), subset_size, replace=False)
        
        X_subset = eeg_data[indices].astype(np.float64)
        y_subset = (labels[indices] > 0).astype(np.int32)  # Binary classification
        
        print(f"✅ Subset created: X{X_subset.shape}, y{y_subset.shape}")
        print(f"✅ Label distribution: {np.bincount(y_subset)}")
        
        # Test with LogisticRegression only (simpler)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_subset)
        
        lr = LogisticRegression(random_state=42)
        lr.fit(X_scaled, y_subset)
        pred = lr.predict(X_scaled)
        
        accuracy = np.mean(pred == y_subset)
        print(f"✅ LogisticRegression accuracy: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔧 Minimal Environment Test")
    print("=" * 30)
    
    # Check files
    files_ok = check_files()
    
    if not files_ok:
        print("\n❌ Required files not found")
        return
    
    # Test data loading
    eeg_data, labels = test_data_loading()
    
    # Test sklearn
    sklearn_ok = test_sklearn()
    
    # Test with real data
    real_data_ok = test_with_real_data(eeg_data, labels)
    
    print(f"\n📊 Test Summary:")
    print(f"Files: {'✅' if files_ok else '❌'}")
    print(f"Data loading: {'✅' if eeg_data is not None else '❌'}")
    print(f"Sklearn: {'✅' if sklearn_ok else '❌'}")
    print(f"Real data: {'✅' if real_data_ok else '❌'}")
    
    if all([files_ok, eeg_data is not None, sklearn_ok, real_data_ok]):
        print(f"\n🎉 All tests passed! Environment is ready.")
    else:
        print(f"\n❌ Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main()

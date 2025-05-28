#!/usr/bin/env python3
# simple_cross_validation.py - Simple cross-subject validation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare handwritten data"""
    print("📂 Loading handwritten data...")
    
    try:
        eeg_data = np.load('handwritten_eeg_data.npy')
        session_labels = np.load('handwritten_session_labels.npy')
        
        print(f"✅ Data loaded: EEG{eeg_data.shape}, Labels{session_labels.shape}")
        print(f"📊 Sessions: {np.bincount(session_labels)}")
        
        return eeg_data, session_labels
        
    except FileNotFoundError:
        print("❌ Handwritten data not found")
        return None, None

def create_subjects(eeg_data, session_labels):
    """Create subject data from sessions"""
    print("🔧 Creating subjects from sessions...")
    
    subjects = {}
    
    for session_id in range(4):
        session_mask = session_labels == session_id
        session_data = eeg_data[session_mask]
        
        if len(session_data) == 0:
            continue
        
        # Subsample for efficiency (every 50th sample)
        subsample_rate = 50
        indices = np.arange(0, len(session_data), subsample_rate)
        subsampled_data = session_data[indices]
        
        # Create binary labels (first half vs second half)
        n_samples = len(subsampled_data)
        split_point = n_samples // 2
        binary_labels = np.zeros(n_samples, dtype=np.int32)
        binary_labels[split_point:] = 1
        
        subjects[session_id] = {
            'data': subsampled_data.astype(np.float64),
            'labels': binary_labels,
            'session_id': session_id,
            'n_samples': n_samples
        }
        
        print(f"  Subject {session_id}: {subsampled_data.shape}, labels: {np.bincount(binary_labels)}")
    
    return subjects

def extract_simple_features(data):
    """Extract simple statistical features"""
    # Use only basic statistical features to avoid complexity
    features = []
    
    # Mean and std for each channel
    features.extend(np.mean(data, axis=0))
    features.extend(np.std(data, axis=0))
    
    return np.array(features, dtype=np.float64)

def prepare_features(subjects):
    """Prepare features for all subjects"""
    print("🔧 Extracting features...")
    
    for subject_id, subject_data in subjects.items():
        eeg_data = subject_data['data']
        
        # Use sliding window
        window_size = 200
        step_size = 100
        
        features_list = []
        labels_list = []
        
        for i in range(0, len(eeg_data) - window_size, step_size):
            window = eeg_data[i:i+window_size]
            label = subject_data['labels'][i + window_size//2]
            
            features = extract_simple_features(window)
            features_list.append(features)
            labels_list.append(label)
        
        if len(features_list) > 0:
            subject_features = np.array(features_list, dtype=np.float64)
            subject_labels = np.array(labels_list, dtype=np.int32)
            
            subjects[subject_id]['features'] = subject_features
            subjects[subject_id]['feature_labels'] = subject_labels
            
            print(f"  Subject {subject_id}: {subject_features.shape} features")

def train_and_test(X_train, y_train, X_test, y_test):
    """Train and test models"""
    # Ensure proper data types
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.int32)
    
    # Clean data
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'LR': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42, C=1.0),
        'RF': RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"    {name}: {accuracy:.4f}")
            
        except Exception as e:
            print(f"    {name}: Failed - {str(e)}")
            results[name] = {'accuracy': 0.0, 'predictions': np.zeros_like(y_test)}
    
    return results

def cross_subject_validation(subjects):
    """Perform cross-subject validation"""
    print("\n🧪 Cross-Subject Validation")
    print("=" * 40)
    
    subject_ids = list(subjects.keys())
    all_results = {}
    
    for test_subject in subject_ids:
        train_subjects = [s for s in subject_ids if s != test_subject]
        
        print(f"\n🔍 Test Subject: {test_subject}")
        print(f"📚 Train Subjects: {train_subjects}")
        
        # Prepare training data
        X_train_list = []
        y_train_list = []
        
        for train_subject in train_subjects:
            if 'features' in subjects[train_subject]:
                X_train_list.append(subjects[train_subject]['features'])
                y_train_list.append(subjects[train_subject]['feature_labels'])
        
        if len(X_train_list) == 0:
            print("  ❌ No training data")
            continue
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Prepare test data
        if 'features' not in subjects[test_subject]:
            print("  ❌ No test data")
            continue
        
        X_test = subjects[test_subject]['features']
        y_test = subjects[test_subject]['feature_labels']
        
        print(f"  📊 Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Train and test
        results = train_and_test(X_train, y_train, X_test, y_test)
        all_results[test_subject] = results
    
    return all_results

def analyze_results(results):
    """Analyze cross-subject validation results"""
    print(f"\n📊 Results Analysis")
    print("=" * 30)
    
    model_accuracies = {}
    
    for subject_id, subject_results in results.items():
        print(f"\nSubject {subject_id}:")
        for model_name, result in subject_results.items():
            if model_name not in model_accuracies:
                model_accuracies[model_name] = []
            
            accuracy = result['accuracy']
            model_accuracies[model_name].append(accuracy)
            print(f"  {model_name}: {accuracy:.4f}")
    
    print(f"\n📊 Overall Performance:")
    for model_name, accuracies in model_accuracies.items():
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"{model_name}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return model_accuracies

def create_visualization(model_accuracies):
    """Create simple visualization"""
    print(f"\n📊 Creating visualization...")
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Model comparison
        models = list(model_accuracies.keys())
        means = [np.mean(model_accuracies[m]) for m in models]
        stds = [np.std(model_accuracies[m]) for m in models]
        
        ax1.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Cross-Subject Validation Results')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy distribution
        all_accs = [acc for accs in model_accuracies.values() for acc in accs]
        ax2.hist(all_accs, bins=10, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(all_accs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_accs):.3f}')
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Accuracy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Handwritten Character EEG Cross-Subject Validation')
        plt.tight_layout()
        plt.savefig('simple_cross_validation_results.png', dpi=300, bbox_inches='tight')
        print("  ✅ Saved as 'simple_cross_validation_results.png'")
        plt.close()
        
    except Exception as e:
        print(f"  ❌ Visualization failed: {str(e)}")

def main():
    """Main function"""
    print("🧪 Simple Cross-Subject Validation")
    print("=" * 50)
    
    # Load data
    eeg_data, session_labels = load_and_prepare_data()
    if eeg_data is None:
        return
    
    # Create subjects
    subjects = create_subjects(eeg_data, session_labels)
    
    # Extract features
    prepare_features(subjects)
    
    # Cross-subject validation
    results = cross_subject_validation(subjects)
    
    if results:
        # Analyze results
        model_accuracies = analyze_results(results)
        
        # Create visualization
        create_visualization(model_accuracies)
        
        # Save results
        np.save('simple_cross_validation_results.npy', results)
        
        print(f"\n✅ Cross-subject validation completed!")
        print(f"📊 Results saved")
        
        # Best performance
        best_model = max(model_accuracies.keys(), 
                        key=lambda x: np.mean(model_accuracies[x]))
        best_acc = np.mean(model_accuracies[best_model])
        
        print(f"\n🏆 Best Model: {best_model} ({best_acc:.4f})")
    
    else:
        print(f"\n❌ Cross-subject validation failed!")

if __name__ == "__main__":
    main()

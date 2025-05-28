#!/usr/bin/env python3
# efficient_cross_validation.py - Efficient cross-subject validation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data_efficiently():
    """Load data with efficient subsampling"""
    print("📂 Loading handwritten data efficiently...")
    
    # Load data
    eeg_data = np.load('handwritten_eeg_data.npy')
    session_labels = np.load('handwritten_session_labels.npy')
    
    print(f"✅ Original data: EEG{eeg_data.shape}, Labels{session_labels.shape}")
    
    # Aggressive subsampling for efficiency (every 1000th sample)
    subsample_rate = 1000
    indices = np.arange(0, len(eeg_data), subsample_rate)
    
    eeg_subsampled = eeg_data[indices]
    labels_subsampled = session_labels[indices]
    
    print(f"✅ Subsampled data: EEG{eeg_subsampled.shape}, Labels{labels_subsampled.shape}")
    print(f"📊 Session distribution: {np.bincount(labels_subsampled)}")
    
    return eeg_subsampled, labels_subsampled

def create_subjects_efficient(eeg_data, session_labels):
    """Create subjects efficiently"""
    print("🔧 Creating subjects...")
    
    subjects = {}
    
    for session_id in range(4):
        session_mask = session_labels == session_id
        session_data = eeg_data[session_mask]
        
        if len(session_data) < 10:  # Skip if too few samples
            print(f"  ⚠️ Session {session_id}: Too few samples ({len(session_data)})")
            continue
        
        # Create binary classification task
        n_samples = len(session_data)
        split_point = n_samples // 2
        
        # Binary labels: first half vs second half
        binary_labels = np.zeros(n_samples, dtype=np.int32)
        binary_labels[split_point:] = 1
        
        subjects[session_id] = {
            'data': session_data.astype(np.float64),
            'labels': binary_labels,
            'name': f'Session_{session_id}',
            'n_samples': n_samples
        }
        
        print(f"  ✅ Session {session_id}: {session_data.shape}, labels: {np.bincount(binary_labels)}")
    
    return subjects

def extract_features_efficient(data):
    """Extract efficient features"""
    # Simple statistical features only
    features = []
    
    # Basic statistics per channel
    features.extend(np.mean(data, axis=0))  # 64 features
    features.extend(np.std(data, axis=0))   # 64 features
    features.extend(np.min(data, axis=0))   # 64 features
    features.extend(np.max(data, axis=0))   # 64 features
    
    return np.array(features, dtype=np.float64)

def prepare_subject_features(subjects):
    """Prepare features for subjects"""
    print("🔧 Extracting features...")
    
    for subject_id, subject_data in subjects.items():
        eeg_data = subject_data['data']
        labels = subject_data['labels']
        
        # Use non-overlapping windows for efficiency
        window_size = 50  # Small window
        
        features_list = []
        labels_list = []
        
        # Non-overlapping windows
        for i in range(0, len(eeg_data) - window_size, window_size):
            window = eeg_data[i:i+window_size]
            label = labels[i + window_size//2]
            
            features = extract_features_efficient(window)
            features_list.append(features)
            labels_list.append(label)
        
        if len(features_list) > 0:
            subject_features = np.array(features_list, dtype=np.float64)
            subject_labels = np.array(labels_list, dtype=np.int32)
            
            # Clean features
            subject_features = np.nan_to_num(subject_features)
            
            subjects[subject_id]['features'] = subject_features
            subjects[subject_id]['feature_labels'] = subject_labels
            
            print(f"  ✅ {subject_data['name']}: {subject_features.shape} features")
            print(f"    Label distribution: {np.bincount(subject_labels)}")

def train_models_safe(X_train, y_train):
    """Train models safely"""
    # Ensure proper data types and clean data
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int32)
    X_train = np.nan_to_num(X_train)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Simple models with conservative parameters
    models = {}
    
    try:
        models['LR'] = LogisticRegression(random_state=42, max_iter=500)
        models['LR'].fit(X_train_scaled, y_train)
        print(f"    ✅ LR trained")
    except Exception as e:
        print(f"    ❌ LR failed: {str(e)}")
    
    try:
        models['SVM'] = SVC(probability=True, random_state=42, C=1.0)
        models['SVM'].fit(X_train_scaled, y_train)
        print(f"    ✅ SVM trained")
    except Exception as e:
        print(f"    ❌ SVM failed: {str(e)}")
    
    try:
        models['RF'] = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=3)
        models['RF'].fit(X_train_scaled, y_train)
        print(f"    ✅ RF trained")
    except Exception as e:
        print(f"    ❌ RF failed: {str(e)}")
    
    # Create ensemble if we have multiple models
    if len(models) > 1:
        try:
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in models.items()],
                voting='soft'
            )
            ensemble.fit(X_train_scaled, y_train)
            models['Ensemble'] = ensemble
            print(f"    ✅ Ensemble trained")
        except Exception as e:
            print(f"    ❌ Ensemble failed: {str(e)}")
    
    return models, scaler

def evaluate_models_safe(models, scaler, X_test, y_test):
    """Evaluate models safely"""
    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.int32)
    X_test = np.nan_to_num(X_test)
    
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"    ✅ {name}: {accuracy:.4f}")
            
        except Exception as e:
            print(f"    ❌ {name}: {str(e)}")
            results[name] = {'accuracy': 0.0, 'predictions': np.zeros_like(y_test)}
    
    return results

def cross_subject_validation_efficient(subjects):
    """Efficient cross-subject validation"""
    print("\n🧪 Efficient Cross-Subject Validation")
    print("=" * 50)
    
    subject_ids = list(subjects.keys())
    all_results = {}
    
    for test_subject in subject_ids:
        train_subjects = [s for s in subject_ids if s != test_subject]
        
        print(f"\n🔍 Test Subject: {subjects[test_subject]['name']}")
        print(f"📚 Train Subjects: {[subjects[s]['name'] for s in train_subjects]}")
        
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
        print(f"  📊 Train labels: {np.bincount(y_train)}, Test labels: {np.bincount(y_test)}")
        
        # Train models
        print(f"  🔧 Training models...")
        models, scaler = train_models_safe(X_train, y_train)
        
        if len(models) == 0:
            print("  ❌ No models trained successfully")
            continue
        
        # Evaluate models
        print(f"  🧪 Evaluating models...")
        results = evaluate_models_safe(models, scaler, X_test, y_test)
        
        all_results[test_subject] = {
            'subject_name': subjects[test_subject]['name'],
            'results': results,
            'train_size': len(y_train),
            'test_size': len(y_test)
        }
    
    return all_results

def analyze_and_visualize(all_results):
    """Analyze results and create visualization"""
    print(f"\n📊 Results Analysis")
    print("=" * 30)
    
    # Collect accuracies
    model_accuracies = {}
    
    for subject_id, subject_results in all_results.items():
        subject_name = subject_results['subject_name']
        print(f"\n{subject_name}:")
        
        for model_name, result in subject_results['results'].items():
            if model_name not in model_accuracies:
                model_accuracies[model_name] = []
            
            accuracy = result['accuracy']
            model_accuracies[model_name].append(accuracy)
            print(f"  {model_name}: {accuracy:.4f}")
    
    # Overall statistics
    print(f"\n📊 Overall Performance:")
    print("-" * 25)
    
    for model_name, accuracies in model_accuracies.items():
        if len(accuracies) > 0:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"{model_name}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    # Create visualization
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Model comparison
        models = list(model_accuracies.keys())
        means = [np.mean(model_accuracies[m]) for m in models]
        stds = [np.std(model_accuracies[m]) for m in models]
        
        axes[0].bar(models, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Performance')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Plot 2: Subject comparison
        subjects = [all_results[sid]['subject_name'] for sid in all_results.keys()]
        
        # Use best model for subject comparison
        best_model = max(model_accuracies.keys(), key=lambda x: np.mean(model_accuracies[x]))
        subject_accs = []
        
        for subject_id in all_results.keys():
            if best_model in all_results[subject_id]['results']:
                acc = all_results[subject_id]['results'][best_model]['accuracy']
                subject_accs.append(acc)
            else:
                subject_accs.append(0)
        
        axes[1].bar(range(len(subjects)), subject_accs, alpha=0.7, color='lightcoral')
        axes[1].set_xlabel('Subject')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Subject Performance ({best_model})')
        axes[1].set_xticks(range(len(subjects)))
        axes[1].set_xticklabels([s.split('_')[1] for s in subjects])
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        # Plot 3: Summary
        axes[2].text(0.1, 0.9, 'Cross-Subject Validation', fontsize=14, fontweight='bold',
                    transform=axes[2].transAxes)
        axes[2].text(0.1, 0.8, f'Subjects: {len(subjects)}', transform=axes[2].transAxes)
        axes[2].text(0.1, 0.7, f'Models: {len(models)}', transform=axes[2].transAxes)
        
        if model_accuracies:
            best_acc = np.mean(model_accuracies[best_model])
            axes[2].text(0.1, 0.6, f'Best Model: {best_model}', transform=axes[2].transAxes)
            axes[2].text(0.1, 0.5, f'Best Accuracy: {best_acc:.4f}', transform=axes[2].transAxes)
        
        axes[2].text(0.1, 0.3, 'Handwritten Character EEG', transform=axes[2].transAxes)
        axes[2].text(0.1, 0.2, 'Temporal Classification Task', transform=axes[2].transAxes)
        
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')
        
        plt.suptitle('Efficient Cross-Subject Validation Results', fontsize=16)
        plt.tight_layout()
        plt.savefig('efficient_cross_validation_results.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Visualization saved as 'efficient_cross_validation_results.png'")
        plt.close()
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {str(e)}")
    
    return model_accuracies

def main():
    """Main function"""
    print("🧪 Efficient Cross-Subject Validation for Handwritten Character EEG")
    print("=" * 70)
    
    try:
        # Load data efficiently
        eeg_data, session_labels = load_data_efficiently()
        
        # Create subjects
        subjects = create_subjects_efficient(eeg_data, session_labels)
        
        if len(subjects) < 2:
            print("❌ Need at least 2 subjects for cross-validation")
            return
        
        # Extract features
        prepare_subject_features(subjects)
        
        # Cross-subject validation
        all_results = cross_subject_validation_efficient(subjects)
        
        if all_results:
            # Analyze and visualize
            model_accuracies = analyze_and_visualize(all_results)
            
            # Save results
            np.save('efficient_cross_validation_results.npy', all_results)
            
            print(f"\n✅ Efficient cross-subject validation completed!")
            print(f"📊 Results saved as 'efficient_cross_validation_results.npy'")
            
            # Best performance summary
            if model_accuracies:
                best_model = max(model_accuracies.keys(), 
                               key=lambda x: np.mean(model_accuracies[x]))
                best_acc = np.mean(model_accuracies[best_model])
                best_std = np.std(model_accuracies[best_model])
                
                print(f"\n🏆 Best Performance:")
                print(f"Model: {best_model}")
                print(f"Accuracy: {best_acc:.4f} ± {best_std:.4f}")
                
                # Interpretation
                if best_acc > 0.7:
                    print(f"✅ Excellent cross-subject generalization!")
                elif best_acc > 0.6:
                    print(f"⚠️ Good generalization with room for improvement")
                else:
                    print(f"❌ Limited generalization - challenging task")
        
        else:
            print(f"\n❌ Cross-subject validation failed!")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

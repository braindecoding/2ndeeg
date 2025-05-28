#!/usr/bin/env python3
# test_cross_subject_synthetic.py - Test cross-subject validation with synthetic data

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def create_synthetic_subjects(n_subjects=5, n_trials_per_class=100):
    """Create synthetic EEG data for multiple subjects with different characteristics"""
    print(f"🔧 Creating synthetic data for {n_subjects} subjects...")
    
    n_channels = 14
    n_timepoints = 128
    n_classes = 2
    
    subjects_data = {}
    
    for subject_id in range(1, n_subjects + 1):
        print(f"  🔧 Creating data for Subject {subject_id}...")
        
        # Set different random seed for each subject to simulate individual differences
        np.random.seed(42 + subject_id * 10)
        
        all_trials = []
        all_labels = []
        
        # Subject-specific characteristics
        subject_noise_level = 0.3 + 0.2 * np.random.random()  # Different noise levels
        subject_amplitude = 0.8 + 0.4 * np.random.random()   # Different signal amplitudes
        subject_frequency = 8 + 4 * np.random.random()       # Different dominant frequencies
        
        for class_id in range(n_classes):
            for trial in range(n_trials_per_class):
                # Create base signal
                base_signal = np.random.randn(n_channels, n_timepoints) * subject_noise_level
                
                # Add class-specific patterns
                time_axis = np.linspace(0, 2*np.pi, n_timepoints)
                
                if class_id == 0:
                    # Class 0: Simulating "digit 6" - more frontal activity
                    frontal_pattern = np.sin(subject_frequency * time_axis) * subject_amplitude
                    base_signal[:4, :] += frontal_pattern  # Frontal channels
                    
                    # Add some alpha activity in occipital
                    alpha_pattern = np.sin(10 * time_axis) * (subject_amplitude * 0.5)
                    base_signal[-2:, :] += alpha_pattern  # Occipital channels
                    
                else:
                    # Class 1: Simulating "digit 9" - more parietal-occipital activity
                    parietal_pattern = np.cos(subject_frequency * time_axis) * subject_amplitude
                    base_signal[6:10, :] += parietal_pattern  # Parietal channels
                    
                    # Add some beta activity in central
                    beta_pattern = np.sin(20 * time_axis) * (subject_amplitude * 0.3)
                    base_signal[4:6, :] += beta_pattern  # Central channels
                
                # Add subject-specific drift
                drift = np.linspace(0, 0.1 * subject_amplitude, n_timepoints)
                base_signal += drift
                
                all_trials.append(base_signal)
                all_labels.append(class_id)
        
        # Convert to numpy arrays
        eeg_data = np.array(all_trials)
        labels = np.array(all_labels)
        
        # Shuffle data
        indices = np.random.permutation(len(eeg_data))
        eeg_data = eeg_data[indices]
        labels = labels[indices]
        
        subjects_data[subject_id] = {
            'eeg_data': eeg_data,
            'labels': labels,
            'characteristics': {
                'noise_level': subject_noise_level,
                'amplitude': subject_amplitude,
                'frequency': subject_frequency
            }
        }
        
        print(f"    ✅ Subject {subject_id}: {eeg_data.shape}, noise={subject_noise_level:.2f}, amp={subject_amplitude:.2f}")
    
    return subjects_data

def extract_features_for_subject(eeg_data):
    """Extract features for a subject's data"""
    print("  🧩 Extracting features...")
    
    # Flatten data for feature extraction
    eeg_data_flat = eeg_data.reshape(len(eeg_data), -1)
    
    # Simple feature extraction (mean, std, etc.)
    features = []
    
    for trial in eeg_data:
        trial_features = []
        
        # Statistical features for each channel
        for channel in range(trial.shape[0]):
            channel_data = trial[channel]
            
            # Basic statistical features
            trial_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.max(channel_data),
                np.min(channel_data),
                np.var(channel_data)
            ])
            
            # Frequency domain features (simplified)
            fft = np.fft.fft(channel_data)
            power_spectrum = np.abs(fft)**2
            
            # Power in different frequency bands (simplified)
            delta_power = np.mean(power_spectrum[1:4])    # ~1-3 Hz
            theta_power = np.mean(power_spectrum[4:8])    # ~4-7 Hz
            alpha_power = np.mean(power_spectrum[8:13])   # ~8-12 Hz
            beta_power = np.mean(power_spectrum[13:30])   # ~13-29 Hz
            
            trial_features.extend([delta_power, theta_power, alpha_power, beta_power])
        
        # Cross-channel features
        # Correlation between frontal channels
        frontal_corr = np.corrcoef(trial[0], trial[1])[0, 1]
        # Correlation between occipital channels
        occipital_corr = np.corrcoef(trial[-2], trial[-1])[0, 1]
        
        trial_features.extend([frontal_corr, occipital_corr])
        
        features.append(trial_features)
    
    features = np.array(features, dtype=np.float64)
    
    # Handle NaN values
    features = np.nan_to_num(features)
    
    print(f"    ✅ Features extracted: {features.shape}")
    return features

def load_trained_models():
    """Load our pre-trained models"""
    print("📂 Loading pre-trained models...")
    
    try:
        # Load traditional models
        traditional_models = joblib.load('traditional_models.pkl')
        
        # Load meta-model
        meta_model = joblib.load('meta_model.pkl')
        
        print("  ✅ Models loaded successfully")
        return traditional_models, meta_model
        
    except Exception as e:
        print(f"  ❌ Error loading models: {str(e)}")
        print("  ℹ️ Make sure you have run ensemble_model.py first to train the models")
        return None, None

def test_subject_with_ensemble(traditional_models, meta_model, features, labels):
    """Test a subject using our ensemble model"""
    print("  🧪 Testing with ensemble model...")
    
    try:
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get predictions from traditional models
        predictions_list = []
        
        if 'svm' in traditional_models and traditional_models['svm'] is not None:
            svm_proba = traditional_models['svm'].predict_proba(features_scaled)
            predictions_list.append(svm_proba)
        
        if 'lr' in traditional_models and traditional_models['lr'] is not None:
            lr_proba = traditional_models['lr'].predict_proba(features_scaled)
            predictions_list.append(lr_proba)
        
        if len(predictions_list) == 0:
            print("    ❌ No valid traditional models found")
            return None
        
        # Combine predictions for meta-model
        if len(predictions_list) == 1:
            meta_features = predictions_list[0]
        else:
            meta_features = np.hstack(predictions_list)
        
        # Get final predictions from meta-model
        final_predictions = meta_model.predict(meta_features)
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, final_predictions)
        
        print(f"    ✅ Accuracy: {accuracy:.4f}")
        
        return accuracy, final_predictions
        
    except Exception as e:
        print(f"    ❌ Error in testing: {str(e)}")
        return None, None

def cross_subject_validation_synthetic():
    """Perform cross-subject validation with synthetic data"""
    print("🚀 Cross-Subject Validation with Synthetic Data")
    print("=" * 50)
    
    # Load trained models
    traditional_models, meta_model = load_trained_models()
    
    if traditional_models is None or meta_model is None:
        print("❌ Cannot proceed without trained models")
        return
    
    # Create synthetic subjects
    subjects_data = create_synthetic_subjects(n_subjects=5, n_trials_per_class=80)
    
    # Test each subject
    subject_accuracies = []
    subject_predictions = {}
    
    for subject_id, subject_data in subjects_data.items():
        print(f"\n🧪 Testing Subject {subject_id}")
        print("-" * 30)
        
        eeg_data = subject_data['eeg_data']
        labels = subject_data['labels']
        characteristics = subject_data['characteristics']
        
        print(f"  📊 Data shape: {eeg_data.shape}")
        print(f"  📊 Class distribution: {np.bincount(labels)}")
        print(f"  📊 Subject characteristics: noise={characteristics['noise_level']:.2f}, "
              f"amp={characteristics['amplitude']:.2f}, freq={characteristics['frequency']:.1f}")
        
        # Extract features
        features = extract_features_for_subject(eeg_data)
        
        # Test with ensemble
        result = test_subject_with_ensemble(traditional_models, meta_model, features, labels)
        
        if result is not None:
            accuracy, predictions = result
            subject_accuracies.append(accuracy)
            subject_predictions[subject_id] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'true_labels': labels,
                'characteristics': characteristics
            }
        else:
            print(f"    ❌ Testing failed for Subject {subject_id}")
    
    # Analysis and summary
    if len(subject_accuracies) > 0:
        print(f"\n📊 Cross-Subject Validation Results")
        print("=" * 40)
        
        mean_accuracy = np.mean(subject_accuracies)
        std_accuracy = np.std(subject_accuracies)
        
        print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Best accuracy: {np.max(subject_accuracies):.4f}")
        print(f"Worst accuracy: {np.min(subject_accuracies):.4f}")
        print(f"Number of subjects: {len(subject_accuracies)}")
        
        # Detailed results for each subject
        print(f"\nDetailed Results:")
        print("-" * 20)
        for subject_id, results in subject_predictions.items():
            accuracy = results['accuracy']
            characteristics = results['characteristics']
            
            print(f"Subject {subject_id}: {accuracy:.4f} "
                  f"(noise={characteristics['noise_level']:.2f}, "
                  f"amp={characteristics['amplitude']:.2f})")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Accuracy by subject
        plt.subplot(2, 2, 1)
        subjects = list(subject_predictions.keys())
        accuracies = [subject_predictions[s]['accuracy'] for s in subjects]
        
        plt.bar(subjects, accuracies)
        plt.axhline(y=mean_accuracy, color='r', linestyle='--', label=f'Mean: {mean_accuracy:.3f}')
        plt.xlabel('Subject ID')
        plt.ylabel('Accuracy')
        plt.title('Cross-Subject Validation Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Accuracy vs noise level
        plt.subplot(2, 2, 2)
        noise_levels = [subject_predictions[s]['characteristics']['noise_level'] for s in subjects]
        plt.scatter(noise_levels, accuracies)
        plt.xlabel('Noise Level')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Noise Level')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Accuracy vs amplitude
        plt.subplot(2, 2, 3)
        amplitudes = [subject_predictions[s]['characteristics']['amplitude'] for s in subjects]
        plt.scatter(amplitudes, accuracies)
        plt.xlabel('Signal Amplitude')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Signal Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Confusion matrix for best subject
        plt.subplot(2, 2, 4)
        best_subject_id = subjects[np.argmax(accuracies)]
        best_predictions = subject_predictions[best_subject_id]['predictions']
        best_labels = subject_predictions[best_subject_id]['true_labels']
        
        cm = confusion_matrix(best_labels, best_predictions)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Best Subject {best_subject_id}')
        plt.colorbar()
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), 
                        horizontalalignment="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        plt.tight_layout()
        plt.savefig('cross_subject_synthetic_validation.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Results plot saved as 'cross_subject_synthetic_validation.png'")
        
        # Save detailed results
        results_summary = {
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'subject_results': subject_predictions,
            'all_accuracies': subject_accuracies
        }
        
        np.save('cross_subject_synthetic_results.npy', results_summary)
        print(f"📊 Detailed results saved as 'cross_subject_synthetic_results.npy'")
        
        return results_summary
    
    else:
        print("❌ No subjects were successfully tested")
        return None

def main():
    """Main function"""
    print("🧪 Cross-Subject Validation Test with Synthetic Data")
    print("=" * 55)
    
    # Run cross-subject validation
    results = cross_subject_validation_synthetic()
    
    if results:
        print("\n✅ Cross-subject validation completed successfully!")
        print(f"📊 Mean accuracy across subjects: {results['mean_accuracy']:.4f}")
        
        # Interpretation
        print(f"\n🔍 Interpretation:")
        if results['mean_accuracy'] > 0.7:
            print("  ✅ Good generalization across subjects")
        elif results['mean_accuracy'] > 0.6:
            print("  ⚠️ Moderate generalization - room for improvement")
        else:
            print("  ❌ Poor generalization - model may be overfitting to original subject")
        
        print(f"\n📝 Next steps:")
        print("  1. Try with real alternative datasets (BCI Competition, PhysioNet)")
        print("  2. Implement domain adaptation techniques if generalization is poor")
        print("  3. Consider subject-specific fine-tuning approaches")
        
    else:
        print("\n❌ Cross-subject validation failed!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# real_dataset_validation.py - Cross-subject validation using real EEG datasets

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import mne
from scipy.signal import resample

def download_physionet_sample():
    """Download a sample of PhysioNet EEG data for testing"""
    print("📥 Downloading PhysioNet EEG sample data...")
    
    import urllib.request
    
    base_url = "https://physionet.org/files/eegmmidb/1.0.0/"
    data_dir = "datasets/physionet_sample"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download data for first 3 subjects, tasks 3 and 7 (left/right hand imagery)
    subjects_to_download = [1, 2, 3]
    tasks_to_download = [3, 7]  # T1 (left fist), T2 (right fist)
    
    downloaded_files = []
    
    for subject_id in subjects_to_download:
        subject_dir = os.path.join(data_dir, f"S{subject_id:03d}")
        os.makedirs(subject_dir, exist_ok=True)
        
        for task_id in tasks_to_download:
            filename = f"S{subject_id:03d}R{task_id:02d}.edf"
            url = base_url + f"S{subject_id:03d}/" + filename
            filepath = os.path.join(subject_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"  📥 Downloading {filename}...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                    print(f"  ✅ Downloaded {filename}")
                    downloaded_files.append(filepath)
                except Exception as e:
                    print(f"  ❌ Failed to download {filename}: {str(e)}")
            else:
                print(f"  ✅ {filename} already exists")
                downloaded_files.append(filepath)
    
    return data_dir, downloaded_files

def load_physionet_subject_data(data_dir, subject_id):
    """Load PhysioNet data for a specific subject"""
    print(f"📂 Loading PhysioNet data for subject {subject_id:03d}...")
    
    try:
        subject_dir = os.path.join(data_dir, f"S{subject_id:03d}")
        
        # Load both task files (left and right hand imagery)
        task_files = [
            os.path.join(subject_dir, f"S{subject_id:03d}R03.edf"),  # Left hand imagery
            os.path.join(subject_dir, f"S{subject_id:03d}R07.edf")   # Right hand imagery
        ]
        
        all_trials = []
        all_labels = []
        
        for task_idx, task_file in enumerate(task_files):
            if not os.path.exists(task_file):
                print(f"  ⚠️ File not found: {task_file}")
                continue
            
            # Load EDF file
            raw = mne.io.read_raw_edf(task_file, preload=True, verbose=False)
            
            # Get data and sampling frequency
            data = raw.get_data()  # Shape: [channels, timepoints]
            sfreq = raw.info['sfreq']
            
            print(f"    📊 Loaded {task_file}: {data.shape}, {sfreq} Hz")
            
            # Extract events from annotations
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            
            # Find motor imagery events (usually event code T1 or T2)
            # For PhysioNet, we look for specific event codes
            target_events = []
            for event in events:
                # Event structure: [sample, prev_id, event_id]
                if event[2] in [2, 3]:  # Motor imagery events
                    target_events.append(event)
            
            print(f"    📊 Found {len(target_events)} motor imagery events")
            
            # Extract trials around events
            trial_duration = 3.0  # 3 seconds
            trial_samples = int(trial_duration * sfreq)
            
            for event in target_events:
                start_sample = event[0]
                end_sample = start_sample + trial_samples
                
                if end_sample <= data.shape[1]:
                    trial_data = data[:, start_sample:end_sample]
                    
                    # Select subset of channels (to match our 14-channel setup)
                    # PhysioNet has 64 channels, we'll select motor cortex related channels
                    selected_channels = np.linspace(0, data.shape[0]-1, 14, dtype=int)
                    trial_data = trial_data[selected_channels, :]
                    
                    # Resample to 128 Hz to match our setup
                    if sfreq != 128:
                        trial_data_resampled = resample(trial_data, 128, axis=1)
                    else:
                        trial_data_resampled = trial_data
                    
                    all_trials.append(trial_data_resampled)
                    all_labels.append(task_idx)  # 0 for left hand, 1 for right hand
        
        if len(all_trials) == 0:
            print(f"  ❌ No valid trials found for subject {subject_id}")
            return None, None
        
        eeg_data = np.array(all_trials)
        labels = np.array(all_labels)
        
        print(f"  ✅ Processed {len(eeg_data)} trials")
        print(f"  📊 Final data shape: {eeg_data.shape}")
        print(f"  📊 Class distribution: {np.bincount(labels)}")
        
        return eeg_data, labels
        
    except Exception as e:
        print(f"  ❌ Error loading PhysioNet data: {str(e)}")
        return None, None

def extract_simple_features(eeg_data):
    """Extract simple but effective features from EEG data"""
    print("  🧩 Extracting features...")
    
    features = []
    
    for trial in eeg_data:
        trial_features = []
        
        # Statistical features for each channel
        for channel in range(trial.shape[0]):
            channel_data = trial[channel]
            
            # Time domain features
            trial_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                np.max(channel_data) - np.min(channel_data),  # Range
                np.percentile(channel_data, 75) - np.percentile(channel_data, 25)  # IQR
            ])
            
            # Simple frequency domain features
            fft = np.fft.fft(channel_data)
            power_spectrum = np.abs(fft[:len(fft)//2])**2
            
            # Power in frequency bands (assuming 128 Hz sampling)
            freqs = np.fft.fftfreq(len(channel_data), 1/128)[:len(fft)//2]
            
            # Define frequency bands
            delta_mask = (freqs >= 1) & (freqs <= 4)
            theta_mask = (freqs >= 4) & (freqs <= 8)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            
            # Calculate band powers
            delta_power = np.mean(power_spectrum[delta_mask]) if np.any(delta_mask) else 0
            theta_power = np.mean(power_spectrum[theta_mask]) if np.any(theta_mask) else 0
            alpha_power = np.mean(power_spectrum[alpha_mask]) if np.any(alpha_mask) else 0
            beta_power = np.mean(power_spectrum[beta_mask]) if np.any(beta_mask) else 0
            
            trial_features.extend([delta_power, theta_power, alpha_power, beta_power])
        
        # Cross-channel features
        # Correlation between hemispheres (simplified)
        left_channels = trial[:7, :]  # First 7 channels (left hemisphere)
        right_channels = trial[7:, :]  # Last 7 channels (right hemisphere)
        
        left_avg = np.mean(left_channels, axis=0)
        right_avg = np.mean(right_channels, axis=0)
        
        hemispheric_corr = np.corrcoef(left_avg, right_avg)[0, 1]
        trial_features.append(hemispheric_corr)
        
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

def test_subject_with_models(traditional_models, meta_model, features, labels):
    """Test a subject using our trained models"""
    print("  🧪 Testing with trained models...")
    
    try:
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get predictions from available traditional models
        predictions_list = []
        
        if 'svm' in traditional_models and traditional_models['svm'] is not None:
            try:
                svm_proba = traditional_models['svm'].predict_proba(features_scaled)
                predictions_list.append(svm_proba)
                print("    ✅ SVM predictions obtained")
            except Exception as e:
                print(f"    ⚠️ SVM prediction failed: {str(e)}")
        
        if 'lr' in traditional_models and traditional_models['lr'] is not None:
            try:
                lr_proba = traditional_models['lr'].predict_proba(features_scaled)
                predictions_list.append(lr_proba)
                print("    ✅ Logistic Regression predictions obtained")
            except Exception as e:
                print(f"    ⚠️ LR prediction failed: {str(e)}")
        
        if len(predictions_list) == 0:
            print("    ❌ No valid predictions from traditional models")
            return None
        
        # Combine predictions for meta-model
        if len(predictions_list) == 1:
            meta_features = predictions_list[0]
        else:
            meta_features = np.hstack(predictions_list)
        
        # Get final predictions from meta-model
        final_predictions = meta_model.predict(meta_features)
        final_proba = meta_model.predict_proba(meta_features)
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, final_predictions)
        
        print(f"    ✅ Accuracy: {accuracy:.4f}")
        
        # Detailed evaluation
        print("    📊 Classification Report:")
        print(classification_report(labels, final_predictions, target_names=['Left Hand', 'Right Hand']))
        
        return accuracy, final_predictions, final_proba
        
    except Exception as e:
        print(f"    ❌ Error in testing: {str(e)}")
        return None, None, None

def cross_subject_validation_real():
    """Perform cross-subject validation with real PhysioNet data"""
    print("🚀 Cross-Subject Validation with Real PhysioNet EEG Data")
    print("=" * 60)
    
    # Download sample data
    data_dir, downloaded_files = download_physionet_sample()
    
    if len(downloaded_files) == 0:
        print("❌ No data files were downloaded")
        return
    
    # Load trained models
    traditional_models, meta_model = load_trained_models()
    
    if traditional_models is None or meta_model is None:
        print("❌ Cannot proceed without trained models")
        return
    
    # Test on each subject
    subjects_to_test = [1, 2, 3]  # First 3 subjects
    subject_results = {}
    
    for subject_id in subjects_to_test:
        print(f"\n🧪 Testing Subject {subject_id:03d}")
        print("-" * 40)
        
        # Load subject data
        eeg_data, labels = load_physionet_subject_data(data_dir, subject_id)
        
        if eeg_data is None:
            print(f"  ⚠️ Skipping subject {subject_id} - data not available")
            continue
        
        # Extract features
        features = extract_simple_features(eeg_data)
        
        # Test with our models
        result = test_subject_with_models(traditional_models, meta_model, features, labels)
        
        if result[0] is not None:
            accuracy, predictions, probabilities = result
            subject_results[subject_id] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'probabilities': probabilities,
                'true_labels': labels,
                'n_trials': len(labels)
            }
        else:
            print(f"    ❌ Testing failed for Subject {subject_id}")
    
    # Analysis and summary
    if len(subject_results) > 0:
        print(f"\n📊 Cross-Subject Validation Results")
        print("=" * 50)
        
        accuracies = [result['accuracy'] for result in subject_results.values()]
        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        
        print(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"Best accuracy: {np.max(accuracies):.4f}")
        print(f"Worst accuracy: {np.min(accuracies):.4f}")
        print(f"Number of subjects tested: {len(subject_results)}")
        
        # Detailed results
        print(f"\nDetailed Results:")
        print("-" * 30)
        for subject_id, result in subject_results.items():
            print(f"Subject {subject_id:03d}: {result['accuracy']:.4f} ({result['n_trials']} trials)")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Accuracy by subject
        plt.subplot(2, 2, 1)
        subjects = list(subject_results.keys())
        subject_accuracies = [subject_results[s]['accuracy'] for s in subjects]
        
        plt.bar([f"S{s:03d}" for s in subjects], subject_accuracies)
        plt.axhline(y=mean_accuracy, color='r', linestyle='--', label=f'Mean: {mean_accuracy:.3f}')
        plt.xlabel('Subject')
        plt.ylabel('Accuracy')
        plt.title('Cross-Subject Validation Results (PhysioNet)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Number of trials per subject
        plt.subplot(2, 2, 2)
        n_trials = [subject_results[s]['n_trials'] for s in subjects]
        plt.bar([f"S{s:03d}" for s in subjects], n_trials)
        plt.xlabel('Subject')
        plt.ylabel('Number of Trials')
        plt.title('Number of Trials per Subject')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Confusion matrix for best subject
        plt.subplot(2, 2, 3)
        best_subject_id = subjects[np.argmax(subject_accuracies)]
        best_predictions = subject_results[best_subject_id]['predictions']
        best_labels = subject_results[best_subject_id]['true_labels']
        
        cm = confusion_matrix(best_labels, best_predictions)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Best Subject S{best_subject_id:03d}')
        plt.colorbar()
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), 
                        horizontalalignment="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Subplot 4: Accuracy distribution
        plt.subplot(2, 2, 4)
        plt.hist(subject_accuracies, bins=5, alpha=0.7, edgecolor='black')
        plt.axvline(x=mean_accuracy, color='r', linestyle='--', label=f'Mean: {mean_accuracy:.3f}')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.title('Accuracy Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cross_subject_physionet_validation.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Results plot saved as 'cross_subject_physionet_validation.png'")
        
        # Save results
        np.save('cross_subject_physionet_results.npy', subject_results)
        print(f"📊 Detailed results saved as 'cross_subject_physionet_results.npy'")
        
        return subject_results
    
    else:
        print("❌ No subjects were successfully tested")
        return None

def main():
    """Main function"""
    print("🧪 Cross-Subject Validation with Real EEG Data")
    print("=" * 50)
    
    # Run cross-subject validation
    results = cross_subject_validation_real()
    
    if results:
        accuracies = [result['accuracy'] for result in results.values()]
        mean_accuracy = np.mean(accuracies)
        
        print("\n✅ Cross-subject validation completed successfully!")
        print(f"📊 Mean accuracy across subjects: {mean_accuracy:.4f}")
        
        # Interpretation
        print(f"\n🔍 Interpretation:")
        if mean_accuracy > 0.7:
            print("  ✅ Good generalization across subjects")
        elif mean_accuracy > 0.6:
            print("  ⚠️ Moderate generalization - room for improvement")
        else:
            print("  ❌ Poor generalization - model may be overfitting to original subject")
        
        print(f"\n📝 Recommendations:")
        if mean_accuracy < 0.6:
            print("  1. Consider domain adaptation techniques")
            print("  2. Implement subject-specific calibration")
            print("  3. Use more robust feature extraction methods")
        else:
            print("  1. Model shows good generalization capability")
            print("  2. Consider testing on more subjects for robust validation")
            print("  3. Explore ensemble methods for further improvement")
        
    else:
        print("\n❌ Cross-subject validation failed!")

if __name__ == "__main__":
    main()

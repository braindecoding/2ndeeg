#!/usr/bin/env python3
# visual_perception_validation.py - Cross-subject validation using visual perception EEG datasets

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import mne
from scipy.signal import resample
import h5py

def download_things_eeg_sample():
    """Information about downloading THINGS-EEG dataset"""
    print("📥 THINGS-EEG Dataset Information")
    print("=" * 40)
    print("🔗 Download Link: https://osf.io/crn2h/")
    print("📄 Paper: https://www.nature.com/articles/s41597-022-01651-8")
    print()
    print("📋 Dataset Structure:")
    print("  - 10 subjects (sub-01 to sub-10)")
    print("  - 63 EEG channels")
    print("  - 1000 Hz sampling rate")
    print("  - 22,248 images from THINGS database")
    print("  - Visual perception paradigm (subjects see real images)")
    print()
    print("📁 Expected file structure after download:")
    print("  datasets/things_eeg/")
    print("  ├── sub-01/")
    print("  │   ├── eeg/")
    print("  │   │   └── sub-01_task-rsvp_eeg.set")
    print("  │   └── beh/")
    print("  ├── sub-02/")
    print("  └── ...")
    print()
    print("⚠️ Note: This is a large dataset (~50GB). Download manually from OSF.")
    
    return "datasets/things_eeg"

def create_visual_perception_demo_data():
    """Create demo data that simulates visual perception EEG"""
    print("🔧 Creating demo visual perception EEG data...")
    
    data_dir = "datasets/visual_perception_demo"
    os.makedirs(data_dir, exist_ok=True)
    
    # Simulate 3 subjects viewing different visual categories
    n_subjects = 3
    n_trials_per_category = 50
    n_channels = 14  # Match our setup
    n_timepoints = 128
    n_categories = 2  # Two visual categories (e.g., faces vs objects)
    
    for subject_id in range(1, n_subjects + 1):
        print(f"  🔧 Creating data for Subject {subject_id}...")
        
        # Set different random seed for each subject
        np.random.seed(42 + subject_id * 100)
        
        all_trials = []
        all_labels = []
        
        # Subject-specific visual response characteristics
        subject_noise = 0.2 + 0.1 * np.random.random()
        subject_amplitude = 1.0 + 0.3 * np.random.random()
        
        for category_id in range(n_categories):
            for trial in range(n_trials_per_category):
                # Create base EEG signal
                base_signal = np.random.randn(n_channels, n_timepoints) * subject_noise
                
                # Add category-specific visual evoked responses
                time_axis = np.linspace(0, 1, n_timepoints)  # 1 second of data
                
                if category_id == 0:
                    # Category 0: Face-like responses (stronger in occipital-temporal)
                    # P100 component around 100ms
                    p100_latency = 0.1  # 100ms
                    p100_idx = int(p100_latency * n_timepoints)
                    p100_response = subject_amplitude * np.exp(-((time_axis - p100_latency) / 0.05)**2)
                    
                    # Stronger in posterior channels (simulating face processing)
                    base_signal[-4:, :] += p100_response * 2
                    
                    # N170 component around 170ms (face-specific)
                    n170_latency = 0.17
                    n170_response = -subject_amplitude * 0.8 * np.exp(-((time_axis - n170_latency) / 0.03)**2)
                    base_signal[-6:-2, :] += n170_response
                    
                else:
                    # Category 1: Object-like responses (more distributed)
                    # P100 component (general visual response)
                    p100_latency = 0.1
                    p100_response = subject_amplitude * 0.7 * np.exp(-((time_axis - p100_latency) / 0.05)**2)
                    base_signal[-4:, :] += p100_response
                    
                    # P300 component around 300ms (object recognition)
                    p300_latency = 0.3
                    p300_response = subject_amplitude * 0.6 * np.exp(-((time_axis - p300_latency) / 0.08)**2)
                    base_signal[4:8, :] += p300_response  # Central-parietal
                
                # Add alpha suppression during visual processing
                alpha_freq = 10  # 10 Hz alpha
                alpha_suppression = -0.3 * subject_amplitude * np.sin(2 * np.pi * alpha_freq * time_axis)
                base_signal[-2:, :] += alpha_suppression  # Occipital alpha
                
                # Add some gamma activity (visual processing)
                gamma_freq = 40  # 40 Hz gamma
                gamma_activity = 0.1 * subject_amplitude * np.sin(2 * np.pi * gamma_freq * time_axis)
                base_signal[-4:, :] += gamma_activity
                
                all_trials.append(base_signal)
                all_labels.append(category_id)
        
        # Convert to numpy arrays
        eeg_data = np.array(all_trials)
        labels = np.array(all_labels)
        
        # Shuffle data
        indices = np.random.permutation(len(eeg_data))
        eeg_data = eeg_data[indices]
        labels = labels[indices]
        
        # Save data
        subject_file = os.path.join(data_dir, f"subject_{subject_id:02d}_visual.npz")
        np.savez(subject_file, 
                eeg_data=eeg_data, 
                labels=labels,
                subject_characteristics={
                    'noise_level': subject_noise,
                    'amplitude': subject_amplitude
                })
        
        print(f"    ✅ Subject {subject_id} visual data saved: {eeg_data.shape}")
        print(f"    📊 Categories: {np.bincount(labels)}")
    
    print(f"  ✅ Demo visual perception dataset created in {data_dir}")
    return data_dir

def load_visual_perception_demo_data(data_path, subject_id):
    """Load demo visual perception dataset"""
    print(f"📂 Loading visual perception data for subject {subject_id}...")
    
    try:
        subject_file = os.path.join(data_path, f"subject_{subject_id:02d}_visual.npz")
        
        if not os.path.exists(subject_file):
            print(f"  ❌ File not found: {subject_file}")
            return None, None
        
        # Load data
        data = np.load(subject_file, allow_pickle=True)
        eeg_data = data['eeg_data']
        labels = data['labels']
        characteristics = data['subject_characteristics'].item()
        
        print(f"  ✅ Loaded {len(eeg_data)} trials for subject {subject_id}")
        print(f"  📊 Data shape: {eeg_data.shape}")
        print(f"  📊 Category distribution: {np.bincount(labels)}")
        print(f"  📊 Subject characteristics: {characteristics}")
        
        return eeg_data, labels
        
    except Exception as e:
        print(f"  ❌ Error loading visual perception data: {str(e)}")
        return None, None

def extract_visual_perception_features(eeg_data):
    """Extract features specific to visual perception EEG"""
    print("  🧩 Extracting visual perception features...")
    
    features = []
    
    for trial in eeg_data:
        trial_features = []
        
        # Time-domain features for each channel
        for channel in range(trial.shape[0]):
            channel_data = trial[channel]
            
            # Basic statistical features
            trial_features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.var(channel_data),
                np.max(channel_data),
                np.min(channel_data),
                np.max(channel_data) - np.min(channel_data),  # Range
            ])
            
            # Event-related potential features (time windows)
            # P100 window (80-120ms) - assuming 128 samples = 1 second
            p100_start, p100_end = int(0.08 * 128), int(0.12 * 128)
            p100_amplitude = np.max(channel_data[p100_start:p100_end]) if p100_end <= len(channel_data) else 0
            
            # N170 window (150-200ms)
            n170_start, n170_end = int(0.15 * 128), int(0.20 * 128)
            n170_amplitude = np.min(channel_data[n170_start:n170_end]) if n170_end <= len(channel_data) else 0
            
            # P300 window (250-400ms)
            p300_start, p300_end = int(0.25 * 128), int(0.40 * 128)
            p300_amplitude = np.max(channel_data[p300_start:p300_end]) if p300_end <= len(channel_data) else 0
            
            trial_features.extend([p100_amplitude, n170_amplitude, p300_amplitude])
            
            # Frequency domain features
            fft = np.fft.fft(channel_data)
            power_spectrum = np.abs(fft[:len(fft)//2])**2
            freqs = np.fft.fftfreq(len(channel_data), 1/128)[:len(fft)//2]
            
            # Visual processing relevant frequency bands
            delta_mask = (freqs >= 1) & (freqs <= 4)
            theta_mask = (freqs >= 4) & (freqs <= 8)
            alpha_mask = (freqs >= 8) & (freqs <= 13)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            gamma_mask = (freqs >= 30) & (freqs <= 50)
            
            # Calculate band powers
            delta_power = np.mean(power_spectrum[delta_mask]) if np.any(delta_mask) else 0
            theta_power = np.mean(power_spectrum[theta_mask]) if np.any(theta_mask) else 0
            alpha_power = np.mean(power_spectrum[alpha_mask]) if np.any(alpha_mask) else 0
            beta_power = np.mean(power_spectrum[beta_mask]) if np.any(beta_mask) else 0
            gamma_power = np.mean(power_spectrum[gamma_mask]) if np.any(gamma_mask) else 0
            
            trial_features.extend([delta_power, theta_power, alpha_power, beta_power, gamma_power])
        
        # Cross-channel features relevant to visual processing
        # Occipital channels (assuming last 4 channels are occipital)
        occipital_channels = trial[-4:, :]
        occipital_avg = np.mean(occipital_channels, axis=0)
        
        # Temporal channels (middle channels)
        temporal_channels = trial[4:8, :]
        temporal_avg = np.mean(temporal_channels, axis=0)
        
        # Frontal channels (first 4 channels)
        frontal_channels = trial[:4, :]
        frontal_avg = np.mean(frontal_channels, axis=0)
        
        # Inter-regional correlations
        occipital_temporal_corr = np.corrcoef(occipital_avg, temporal_avg)[0, 1]
        frontal_occipital_corr = np.corrcoef(frontal_avg, occipital_avg)[0, 1]
        
        trial_features.extend([occipital_temporal_corr, frontal_occipital_corr])
        
        # Global field power (measure of overall brain activity)
        gfp = np.std(trial, axis=0)
        gfp_features = [np.mean(gfp), np.max(gfp), np.std(gfp)]
        trial_features.extend(gfp_features)
        
        features.append(trial_features)
    
    features = np.array(features, dtype=np.float64)
    
    # Handle NaN values
    features = np.nan_to_num(features)
    
    print(f"    ✅ Visual perception features extracted: {features.shape}")
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

def test_visual_perception_subject(traditional_models, meta_model, features, labels):
    """Test a subject using our trained models on visual perception data"""
    print("  🧪 Testing visual perception classification...")
    
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
        
        print(f"    ✅ Visual perception classification accuracy: {accuracy:.4f}")
        
        # Detailed evaluation
        print("    📊 Classification Report:")
        print(classification_report(labels, final_predictions, target_names=['Category 0', 'Category 1']))
        
        return accuracy, final_predictions, final_proba
        
    except Exception as e:
        print(f"    ❌ Error in visual perception testing: {str(e)}")
        return None, None, None

def cross_subject_visual_perception_validation():
    """Perform cross-subject validation with visual perception data"""
    print("🚀 Cross-Subject Validation with Visual Perception EEG Data")
    print("=" * 60)
    
    # Show information about real datasets
    things_eeg_path = download_things_eeg_sample()
    
    print("\n" + "="*60)
    print("For this demo, we'll use simulated visual perception data.")
    print("In practice, you would use real datasets like THINGS-EEG.")
    print("="*60)
    
    # Create demo visual perception data
    data_dir = create_visual_perception_demo_data()
    
    # Load trained models
    traditional_models, meta_model = load_trained_models()
    
    if traditional_models is None or meta_model is None:
        print("❌ Cannot proceed without trained models")
        return
    
    # Test on each subject
    subjects_to_test = [1, 2, 3]
    subject_results = {}
    
    for subject_id in subjects_to_test:
        print(f"\n🧪 Testing Subject {subject_id:02d} (Visual Perception)")
        print("-" * 50)
        
        # Load subject data
        eeg_data, labels = load_visual_perception_demo_data(data_dir, subject_id)
        
        if eeg_data is None:
            print(f"  ⚠️ Skipping subject {subject_id} - data not available")
            continue
        
        # Extract visual perception specific features
        features = extract_visual_perception_features(eeg_data)
        
        # Test with our models
        result = test_visual_perception_subject(traditional_models, meta_model, features, labels)
        
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
        print(f"\n📊 Cross-Subject Visual Perception Validation Results")
        print("=" * 60)
        
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
            print(f"Subject {subject_id:02d}: {result['accuracy']:.4f} ({result['n_trials']} trials)")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Accuracy by subject
        plt.subplot(2, 2, 1)
        subjects = list(subject_results.keys())
        subject_accuracies = [subject_results[s]['accuracy'] for s in subjects]
        
        plt.bar([f"S{s:02d}" for s in subjects], subject_accuracies)
        plt.axhline(y=mean_accuracy, color='r', linestyle='--', label=f'Mean: {mean_accuracy:.3f}')
        plt.xlabel('Subject')
        plt.ylabel('Accuracy')
        plt.title('Cross-Subject Validation (Visual Perception)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Comparison with original digit classification
        plt.subplot(2, 2, 2)
        original_accuracy = 0.83  # Our ensemble model accuracy on digit data
        comparison_data = ['Original\n(Digit)', 'Visual\nPerception']
        comparison_acc = [original_accuracy, mean_accuracy]
        
        plt.bar(comparison_data, comparison_acc, color=['blue', 'orange'])
        plt.ylabel('Accuracy')
        plt.title('Model Generalization Comparison')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Confusion matrix for best subject
        plt.subplot(2, 2, 3)
        best_subject_id = subjects[np.argmax(subject_accuracies)]
        best_predictions = subject_results[best_subject_id]['predictions']
        best_labels = subject_results[best_subject_id]['true_labels']
        
        cm = confusion_matrix(best_labels, best_predictions)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Best Subject S{best_subject_id:02d}')
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
        plt.savefig('cross_subject_visual_perception_validation.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Results plot saved as 'cross_subject_visual_perception_validation.png'")
        
        # Save results
        np.save('cross_subject_visual_perception_results.npy', subject_results)
        print(f"📊 Detailed results saved as 'cross_subject_visual_perception_results.npy'")
        
        return subject_results
    
    else:
        print("❌ No subjects were successfully tested")
        return None

def main():
    """Main function"""
    print("🧪 Cross-Subject Validation with Visual Perception EEG Data")
    print("=" * 60)
    
    # Run cross-subject validation
    results = cross_subject_visual_perception_validation()
    
    if results:
        accuracies = [result['accuracy'] for result in results.values()]
        mean_accuracy = np.mean(accuracies)
        
        print("\n✅ Cross-subject validation completed successfully!")
        print(f"📊 Mean accuracy across subjects: {mean_accuracy:.4f}")
        
        # Interpretation for visual perception
        print(f"\n🔍 Visual Perception Classification Analysis:")
        if mean_accuracy > 0.7:
            print("  ✅ Good generalization for visual perception tasks")
            print("  💡 Model successfully adapts from digit imagery to visual perception")
        elif mean_accuracy > 0.6:
            print("  ⚠️ Moderate generalization - some domain gap exists")
            print("  💡 Consider domain adaptation techniques")
        else:
            print("  ❌ Poor generalization - significant domain gap")
            print("  💡 May need task-specific training or transfer learning")
        
        print(f"\n📝 Recommendations for Visual Perception Research:")
        print("  1. Use real visual perception datasets (THINGS-EEG, EEG-ImageNet)")
        print("  2. Implement domain adaptation between digit imagery and visual perception")
        print("  3. Consider multi-task learning approaches")
        print("  4. Explore visual feature extraction specific to perception tasks")
        
    else:
        print("\n❌ Cross-subject validation failed!")

if __name__ == "__main__":
    main()

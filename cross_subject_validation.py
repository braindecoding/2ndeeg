#!/usr/bin/env python3
# cross_subject_validation.py - Cross-subject validation using alternative datasets

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import torch
import torch.nn as nn
from scipy.io import loadmat
import mne

# Load our trained models
def load_trained_models():
    """Load our pre-trained models"""
    print("📂 Loading pre-trained models...")
    
    try:
        # Load traditional models
        traditional_models = joblib.load('traditional_models.pkl')
        
        # Load meta-model
        meta_model = joblib.load('meta_model.pkl')
        
        # Load deep learning model (need to recreate architecture first)
        from hybrid_cnn_lstm_attention import HybridCNNLSTMAttention
        
        # Load advanced wavelet features to get dimension
        sample_features = np.load("advanced_wavelet_features.npy")
        wavelet_dim = sample_features.shape[1]
        
        dl_model = HybridCNNLSTMAttention(
            input_channels=14,
            seq_length=128,
            wavelet_features_dim=wavelet_dim,
            num_classes=2
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dl_model.load_state_dict(torch.load('dl_model.pth', map_location=device))
        dl_model.to(device)
        dl_model.eval()
        
        print("  ✅ All models loaded successfully")
        return traditional_models, dl_model, meta_model, device
        
    except Exception as e:
        print(f"  ❌ Error loading models: {str(e)}")
        return None, None, None, None

def load_bci_competition_data(data_path, subject_id):
    """Load BCI Competition IV Dataset 2a for specific subject"""
    print(f"📂 Loading BCI Competition data for subject {subject_id}...")
    
    try:
        # Load .mat file for specific subject
        mat_file = f"{data_path}/A0{subject_id}T.mat"
        
        if not os.path.exists(mat_file):
            print(f"  ❌ File not found: {mat_file}")
            return None, None
        
        # Load data
        data = loadmat(mat_file)
        
        # Extract EEG data and labels
        # Note: Actual structure depends on the dataset format
        # This is a template that needs to be adapted
        
        eeg_data = data['data']  # Shape: [trials, channels, timepoints]
        labels = data['labels']  # Shape: [trials]
        
        # Convert to our format (select 2 classes only)
        # Class 1 (left hand) -> 0, Class 2 (right hand) -> 1
        mask = np.isin(labels, [1, 2])
        eeg_data = eeg_data[mask]
        labels = labels[mask]
        labels = (labels - 1).astype(int)  # Convert to 0, 1
        
        print(f"  ✅ Loaded {len(eeg_data)} trials for subject {subject_id}")
        print(f"  📊 Data shape: {eeg_data.shape}")
        print(f"  📊 Class distribution: {np.bincount(labels)}")
        
        return eeg_data, labels
        
    except Exception as e:
        print(f"  ❌ Error loading BCI data: {str(e)}")
        return None, None

def load_physionet_data(data_path, subject_id, task_ids=[1, 2]):
    """Load PhysioNet EEG Motor Movement/Imagery Dataset"""
    print(f"📂 Loading PhysioNet data for subject {subject_id:03d}...")
    
    try:
        # Load EDF files for specific subject and tasks
        eeg_data_list = []
        labels_list = []
        
        for task_id in task_ids:
            edf_file = f"{data_path}/S{subject_id:03d}/S{subject_id:03d}R{task_id:02d}.edf"
            
            if not os.path.exists(edf_file):
                print(f"  ⚠️ File not found: {edf_file}")
                continue
            
            # Load EDF file using MNE
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # Get EEG data
            data = raw.get_data()  # Shape: [channels, timepoints]
            
            # Extract events/annotations for labels
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            
            # Segment data into trials based on events
            # This is a simplified version - actual implementation depends on task structure
            trial_length = 640  # 5 seconds at 128 Hz
            
            for event in events:
                start_sample = event[0]
                end_sample = start_sample + trial_length
                
                if end_sample <= data.shape[1]:
                    trial_data = data[:, start_sample:end_sample]
                    eeg_data_list.append(trial_data)
                    labels_list.append(task_id - 1)  # Convert to 0, 1
        
        if len(eeg_data_list) == 0:
            print(f"  ❌ No valid trials found for subject {subject_id}")
            return None, None
        
        eeg_data = np.array(eeg_data_list)
        labels = np.array(labels_list)
        
        print(f"  ✅ Loaded {len(eeg_data)} trials for subject {subject_id}")
        print(f"  📊 Data shape: {eeg_data.shape}")
        print(f"  📊 Class distribution: {np.bincount(labels)}")
        
        return eeg_data, labels
        
    except Exception as e:
        print(f"  ❌ Error loading PhysioNet data: {str(e)}")
        return None, None

def preprocess_alternative_data(eeg_data, labels):
    """Preprocess alternative dataset to match our format"""
    print("🔄 Preprocessing alternative dataset...")
    
    try:
        # Ensure data is in correct format [trials, channels, timepoints]
        if eeg_data.ndim != 3:
            print(f"  ❌ Unexpected data shape: {eeg_data.shape}")
            return None, None, None
        
        n_trials, n_channels, n_timepoints = eeg_data.shape
        
        # Downsample channels if necessary (to match our 14-channel setup)
        if n_channels > 14:
            # Select subset of channels (preferably motor cortex related)
            # This is a simplified selection - should be based on channel names
            channel_indices = np.linspace(0, n_channels-1, 14, dtype=int)
            eeg_data = eeg_data[:, channel_indices, :]
            print(f"  📊 Downsampled channels from {n_channels} to 14")
        
        # Resample timepoints if necessary (to match our 128 timepoints)
        if n_timepoints != 128:
            from scipy.signal import resample
            eeg_data_resampled = []
            
            for trial in eeg_data:
                trial_resampled = resample(trial, 128, axis=1)
                eeg_data_resampled.append(trial_resampled)
            
            eeg_data = np.array(eeg_data_resampled)
            print(f"  📊 Resampled timepoints from {n_timepoints} to 128")
        
        # Flatten data for traditional ML models (same as our original preprocessing)
        eeg_data_flat = eeg_data.reshape(n_trials, -1)  # [trials, channels*timepoints]
        
        # Extract advanced wavelet features using our existing function
        from advanced_wavelet_features import extract_advanced_wavelet_features
        
        # Reshape to match our format
        reshaped_data = eeg_data  # Already in correct format
        
        # Extract wavelet features
        wavelet_features, _ = extract_advanced_wavelet_features(eeg_data_flat)
        
        print(f"  ✅ Preprocessing completed")
        print(f"  📊 Final data shape: {eeg_data.shape}")
        print(f"  📊 Wavelet features shape: {wavelet_features.shape}")
        
        return reshaped_data, wavelet_features, labels
        
    except Exception as e:
        print(f"  ❌ Error in preprocessing: {str(e)}")
        return None, None, None

def test_models_on_alternative_data(models, dl_model, meta_model, device, 
                                  reshaped_data, wavelet_features, labels):
    """Test our trained models on alternative dataset"""
    print("🧪 Testing models on alternative dataset...")
    
    try:
        traditional_models, dl_model, meta_model, device = models
        
        # Prepare data for traditional models
        X_combined = np.hstack((reshaped_data.reshape(len(reshaped_data), -1), wavelet_features))
        
        # Standardize features (using same scaler as training - this is simplified)
        scaler = StandardScaler()
        X_combined_scaled = scaler.fit_transform(X_combined)
        
        # Get predictions from traditional models
        svm_proba = traditional_models['svm'].predict_proba(X_combined_scaled)
        lr_proba = traditional_models['lr'].predict_proba(X_combined_scaled)
        
        # Get predictions from deep learning model
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        
        # Prepare data for DL model
        X_raw_tensor = torch.FloatTensor(reshaped_data).unsqueeze(1)  # Add channel dimension
        X_wavelet_tensor = torch.FloatTensor(wavelet_features)
        
        dataset = TensorDataset(X_raw_tensor, X_wavelet_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        dl_proba = []
        with torch.no_grad():
            for inputs_raw, inputs_wavelet in dataloader:
                inputs_raw, inputs_wavelet = inputs_raw.to(device), inputs_wavelet.to(device)
                outputs = dl_model(inputs_raw, inputs_wavelet)
                probs = F.softmax(outputs, dim=1)
                dl_proba.extend(probs.cpu().numpy())
        
        dl_proba = np.array(dl_proba)
        
        # Combine predictions for meta-model
        meta_features = np.hstack((svm_proba, lr_proba, dl_proba))
        
        # Get final predictions from meta-model
        final_predictions = meta_model.predict(meta_features)
        final_proba = meta_model.predict_proba(meta_features)
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, final_predictions)
        
        print(f"  ✅ Cross-subject validation accuracy: {accuracy:.4f}")
        
        # Detailed evaluation
        print("\n📊 Detailed Evaluation:")
        print(classification_report(labels, final_predictions, target_names=['Class 0', 'Class 1']))
        
        # Confusion matrix
        cm = confusion_matrix(labels, final_predictions)
        print(f"  Confusion Matrix:")
        print(f"  {cm[0][0]:4d} {cm[0][1]:4d} | Class 0")
        print(f"  {cm[1][0]:4d} {cm[1][1]:4d} | Class 1")
        print(f"    0    1   <- Predicted")
        
        return accuracy, final_predictions, final_proba
        
    except Exception as e:
        print(f"  ❌ Error in model testing: {str(e)}")
        return None, None, None

def cross_subject_validation_pipeline(dataset_type='bci_competition', data_path=None):
    """Complete cross-subject validation pipeline"""
    print("🚀 Cross-Subject Validation Pipeline")
    print("=" * 50)
    
    # Load our trained models
    models_data = load_trained_models()
    if models_data[0] is None:
        print("❌ Failed to load trained models")
        return
    
    traditional_models, dl_model, meta_model, device = models_data
    
    # Test on multiple subjects
    subject_accuracies = []
    
    if dataset_type == 'bci_competition':
        subject_range = range(1, 10)  # Subjects 1-9
        load_func = load_bci_competition_data
    elif dataset_type == 'physionet':
        subject_range = range(1, 21)  # Test on first 20 subjects
        load_func = load_physionet_data
    else:
        print(f"❌ Unknown dataset type: {dataset_type}")
        return
    
    for subject_id in subject_range:
        print(f"\n🧪 Testing on Subject {subject_id}")
        print("-" * 30)
        
        # Load data for this subject
        if dataset_type == 'bci_competition':
            eeg_data, labels = load_func(data_path, subject_id)
        else:
            eeg_data, labels = load_func(data_path, subject_id)
        
        if eeg_data is None:
            print(f"  ⚠️ Skipping subject {subject_id} - data not available")
            continue
        
        # Preprocess data
        reshaped_data, wavelet_features, labels = preprocess_alternative_data(eeg_data, labels)
        
        if reshaped_data is None:
            print(f"  ⚠️ Skipping subject {subject_id} - preprocessing failed")
            continue
        
        # Test models
        accuracy, predictions, probabilities = test_models_on_alternative_data(
            (traditional_models, dl_model, meta_model, device),
            dl_model, meta_model, device,
            reshaped_data, wavelet_features, labels
        )
        
        if accuracy is not None:
            subject_accuracies.append(accuracy)
            print(f"  ✅ Subject {subject_id} accuracy: {accuracy:.4f}")
        else:
            print(f"  ❌ Subject {subject_id} testing failed")
    
    # Summary
    if len(subject_accuracies) > 0:
        mean_accuracy = np.mean(subject_accuracies)
        std_accuracy = np.std(subject_accuracies)
        
        print(f"\n📊 Cross-Subject Validation Summary:")
        print(f"  Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  Best accuracy: {np.max(subject_accuracies):.4f}")
        print(f"  Worst accuracy: {np.min(subject_accuracies):.4f}")
        print(f"  Number of subjects tested: {len(subject_accuracies)}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(subject_accuracies)+1), subject_accuracies)
        plt.axhline(y=mean_accuracy, color='r', linestyle='--', label=f'Mean: {mean_accuracy:.3f}')
        plt.xlabel('Subject')
        plt.ylabel('Accuracy')
        plt.title('Cross-Subject Validation Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('cross_subject_validation_results.png')
        print(f"  📊 Results plot saved as 'cross_subject_validation_results.png'")
        
        return subject_accuracies
    else:
        print("❌ No subjects were successfully tested")
        return None

def main():
    """Main function"""
    # Example usage
    print("Select dataset for cross-subject validation:")
    print("1. BCI Competition IV Dataset 2a")
    print("2. PhysioNet EEG Motor Movement/Imagery Dataset")
    
    # For demonstration, we'll use BCI Competition
    # In practice, you need to download and specify the data path
    
    data_path = "path/to/bci_competition_data"  # Update this path
    
    # Run cross-subject validation
    results = cross_subject_validation_pipeline(
        dataset_type='bci_competition',
        data_path=data_path
    )
    
    if results:
        print("\n✅ Cross-subject validation completed successfully!")
    else:
        print("\n❌ Cross-subject validation failed!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# things_eeg_loader.py - THINGS-EEG Dataset loader and processor

import numpy as np
import os
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import matplotlib.pyplot as plt

class THINGSEEGDataset:
    """THINGS-EEG Dataset loader and processor"""
    
    def __init__(self, data_path="datasets/things_eeg"):
        self.data_path = data_path
        self.subjects = list(range(1, 11))  # 10 subjects
        self.sampling_rate = 1000  # Hz
        self.n_channels = 63
        self.target_sampling_rate = 128  # Downsample to match our models
        self.target_channels = 14  # Downsample channels to match our models
        
    def download_info(self):
        """Display download information for THINGS-EEG Dataset"""
        print("📥 THINGS-EEG Dataset Download Information")
        print("=" * 50)
        print("🔗 Official Links:")
        print("  Paper: https://www.nature.com/articles/s41597-022-01651-8")
        print("  Data: https://osf.io/crn2h/")
        print("  GitHub: https://github.com/ViCCo-Group/THINGS-EEG")
        print()
        print("📋 Dataset Characteristics:")
        print("  - 10 subjects viewing real images")
        print("  - 22,248 images from THINGS database")
        print("  - 63 EEG channels")
        print("  - 1000 Hz sampling rate")
        print("  - Visual perception paradigm")
        print("  - High-quality preprocessed data")
        print()
        print("💾 Download Instructions:")
        print("  1. Go to: https://osf.io/crn2h/")
        print("  2. Download individual subjects or full dataset")
        print("  3. Extract to 'datasets/things_eeg/'")
        print("  4. Expected structure:")
        print("     datasets/things_eeg/")
        print("     ├── sub-01/")
        print("     │   ├── eeg/")
        print("     │   │   └── sub-01_task-rsvp_eeg.set")
        print("     │   └── beh/")
        print("     │       └── sub-01_task-rsvp_beh.tsv")
        print("     ├── sub-02/")
        print("     └── ...")
        print()
        print("⚡ Quick Start Recommendation:")
        print("  - Download 3-4 subjects first (~20GB)")
        print("  - Test implementation")
        print("  - Download more subjects if needed")
        
    def validate_dataset(self):
        """Validate THINGS-EEG dataset structure"""
        print("🔍 Validating THINGS-EEG dataset...")
        
        if not os.path.exists(self.data_path):
            print(f"❌ Dataset path not found: {self.data_path}")
            print("💡 Please download THINGS-EEG dataset first")
            return False
        
        available_subjects = []
        
        for subject_id in self.subjects:
            subject_dir = os.path.join(self.data_path, f"sub-{subject_id:02d}")
            
            if os.path.exists(subject_dir):
                # Check for EEG file
                eeg_file = os.path.join(subject_dir, "eeg", f"sub-{subject_id:02d}_task-rsvp_eeg.set")
                beh_file = os.path.join(subject_dir, "beh", f"sub-{subject_id:02d}_task-rsvp_beh.tsv")
                
                if os.path.exists(eeg_file):
                    available_subjects.append(subject_id)
                    print(f"  ✅ Subject {subject_id:02d}: EEG file found")
                    
                    if os.path.exists(beh_file):
                        print(f"    ✅ Behavioral file found")
                    else:
                        print(f"    ⚠️ Behavioral file missing")
                else:
                    print(f"  ❌ Subject {subject_id:02d}: EEG file missing")
            else:
                print(f"  ❌ Subject {subject_id:02d}: Directory not found")
        
        print(f"\n📊 Available subjects: {available_subjects}")
        print(f"📊 Total subjects found: {len(available_subjects)}")
        
        if len(available_subjects) >= 2:
            print("✅ Sufficient subjects for cross-subject validation")
            return True, available_subjects
        else:
            print("❌ Insufficient subjects for cross-subject validation")
            return False, available_subjects
    
    def load_subject_data(self, subject_id, max_trials=500):
        """Load THINGS-EEG data for a specific subject"""
        print(f"📂 Loading THINGS-EEG subject {subject_id:02d}...")
        
        try:
            # File paths
            eeg_file = os.path.join(self.data_path, f"sub-{subject_id:02d}", "eeg", 
                                   f"sub-{subject_id:02d}_task-rsvp_eeg.set")
            beh_file = os.path.join(self.data_path, f"sub-{subject_id:02d}", "beh", 
                                   f"sub-{subject_id:02d}_task-rsvp_beh.tsv")
            
            if not os.path.exists(eeg_file):
                print(f"  ❌ EEG file not found: {eeg_file}")
                return None, None, None
            
            # Load EEG data using MNE
            print("  📖 Loading EEG data...")
            raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
            
            # Get basic info
            print(f"    📊 Original sampling rate: {raw.info['sfreq']} Hz")
            print(f"    📊 Original channels: {len(raw.ch_names)}")
            print(f"    📊 Duration: {raw.times[-1]:.1f} seconds")
            
            # Load behavioral data
            if os.path.exists(beh_file):
                print("  📖 Loading behavioral data...")
                beh_data = pd.read_csv(beh_file, sep='\t')
                print(f"    📊 Behavioral trials: {len(beh_data)}")
            else:
                print("  ⚠️ Behavioral file not found, using events from EEG")
                beh_data = None
            
            # Extract events
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            print(f"    📊 EEG events found: {len(events)}")
            
            # Process trials
            trials_data, trials_labels = self._extract_trials(raw, events, beh_data, max_trials)
            
            if trials_data is None:
                return None, None, None
            
            print(f"  ✅ Processed {len(trials_data)} trials")
            print(f"  📊 Final data shape: {trials_data.shape}")
            
            return trials_data, trials_labels, beh_data
            
        except Exception as e:
            print(f"  ❌ Error loading THINGS-EEG data: {str(e)}")
            return None, None, None
    
    def _extract_trials(self, raw, events, beh_data, max_trials):
        """Extract trials from continuous EEG data"""
        print("  🔧 Extracting trials...")
        
        try:
            # Define trial parameters
            tmin, tmax = -0.1, 0.5  # 100ms before to 500ms after stimulus
            baseline = (None, 0)
            
            # Create epochs
            epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax,
                               baseline=baseline, preload=True, verbose=False)
            
            print(f"    📊 Created {len(epochs)} epochs")
            
            # Get epoch data
            epochs_data = epochs.get_data()  # Shape: [n_epochs, n_channels, n_times]
            
            # Downsample channels (select subset to match our 14-channel setup)
            if epochs_data.shape[1] > self.target_channels:
                # Select channels distributed across the scalp
                channel_indices = np.linspace(0, epochs_data.shape[1]-1, self.target_channels, dtype=int)
                epochs_data = epochs_data[:, channel_indices, :]
                print(f"    📊 Downsampled channels: {epochs_data.shape[1]}")
            
            # Downsample time points (to match our 128 timepoints)
            if epochs_data.shape[2] != 128:
                epochs_data_resampled = []
                for epoch in epochs_data:
                    epoch_resampled = resample(epoch, 128, axis=1)
                    epochs_data_resampled.append(epoch_resampled)
                epochs_data = np.array(epochs_data_resampled)
                print(f"    📊 Resampled timepoints: {epochs_data.shape[2]}")
            
            # Limit number of trials
            if len(epochs_data) > max_trials:
                epochs_data = epochs_data[:max_trials]
                print(f"    📊 Limited to {max_trials} trials")
            
            # Create simple labels (for binary classification)
            # We'll create artificial binary labels based on trial index
            # In real implementation, you would use image categories
            labels = np.array([i % 2 for i in range(len(epochs_data))])
            
            print(f"    📊 Label distribution: {np.bincount(labels)}")
            
            return epochs_data, labels
            
        except Exception as e:
            print(f"    ❌ Error extracting trials: {str(e)}")
            return None, None
    
    def extract_visual_features(self, eeg_data):
        """Extract features optimized for visual perception tasks"""
        print("  🧩 Extracting visual perception features...")
        
        features = []
        
        for trial in eeg_data:
            trial_features = []
            
            # Enhanced feature extraction for visual processing
            for channel in range(trial.shape[0]):
                channel_data = trial[channel]
                
                # Basic statistical features
                trial_features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.var(channel_data),
                    np.max(channel_data),
                    np.min(channel_data),
                    np.median(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                    np.max(channel_data) - np.min(channel_data)  # Range
                ])
                
                # Visual ERP components (assuming 128 timepoints = 600ms)
                # P100 window (80-120ms)
                p100_start, p100_end = int(0.08/0.6 * 128), int(0.12/0.6 * 128)
                p100_mean = np.mean(channel_data[p100_start:p100_end]) if p100_end <= len(channel_data) else 0
                p100_max = np.max(channel_data[p100_start:p100_end]) if p100_end <= len(channel_data) else 0
                
                # N170 window (150-200ms) - visual processing
                n170_start, n170_end = int(0.15/0.6 * 128), int(0.20/0.6 * 128)
                n170_mean = np.mean(channel_data[n170_start:n170_end]) if n170_end <= len(channel_data) else 0
                n170_min = np.min(channel_data[n170_start:n170_end]) if n170_end <= len(channel_data) else 0
                
                # P300 window (250-400ms) - recognition
                p300_start, p300_end = int(0.25/0.6 * 128), int(0.40/0.6 * 128)
                p300_mean = np.mean(channel_data[p300_start:p300_end]) if p300_end <= len(channel_data) else 0
                p300_max = np.max(channel_data[p300_start:p300_end]) if p300_end <= len(channel_data) else 0
                
                trial_features.extend([p100_mean, p100_max, n170_mean, n170_min, p300_mean, p300_max])
                
                # Frequency domain features
                fft = np.fft.fft(channel_data)
                power_spectrum = np.abs(fft[:len(fft)//2])**2
                freqs = np.fft.fftfreq(len(channel_data), 1/128)[:len(fft)//2]
                
                # Visual processing frequency bands
                delta_mask = (freqs >= 1) & (freqs <= 4)
                theta_mask = (freqs >= 4) & (freqs <= 8)
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                beta_mask = (freqs >= 13) & (freqs <= 30)
                gamma_mask = (freqs >= 30) & (freqs <= 50)
                
                # Band powers
                delta_power = np.mean(power_spectrum[delta_mask]) if np.any(delta_mask) else 0
                theta_power = np.mean(power_spectrum[theta_mask]) if np.any(theta_mask) else 0
                alpha_power = np.mean(power_spectrum[alpha_mask]) if np.any(alpha_mask) else 0
                beta_power = np.mean(power_spectrum[beta_mask]) if np.any(beta_mask) else 0
                gamma_power = np.mean(power_spectrum[gamma_mask]) if np.any(gamma_mask) else 0
                
                trial_features.extend([delta_power, theta_power, alpha_power, beta_power, gamma_power])
            
            # Cross-channel features for visual processing
            # Posterior channels (visual cortex)
            posterior_channels = trial[-4:, :]  # Last 4 channels (posterior)
            anterior_channels = trial[:4, :]    # First 4 channels (anterior)
            
            # Regional averages
            posterior_avg = np.mean(posterior_channels, axis=0)
            anterior_avg = np.mean(anterior_channels, axis=0)
            
            # Inter-regional correlation
            if len(posterior_avg) > 0 and len(anterior_avg) > 0:
                posterior_anterior_corr = np.corrcoef(posterior_avg, anterior_avg)[0, 1]
            else:
                posterior_anterior_corr = 0
            
            trial_features.append(posterior_anterior_corr)
            
            # Global field power (measure of overall brain activity)
            gfp = np.std(trial, axis=0)
            gfp_mean = np.mean(gfp)
            gfp_max = np.max(gfp)
            gfp_std = np.std(gfp)
            
            trial_features.extend([gfp_mean, gfp_max, gfp_std])
            
            features.append(trial_features)
        
        features = np.array(features, dtype=np.float64)
        
        # Handle NaN values
        features = np.nan_to_num(features)
        
        print(f"    ✅ Visual perception features extracted: {features.shape}")
        return features
    
    def visualize_subject_data(self, subject_id, eeg_data, labels, n_examples=4):
        """Visualize EEG data for a subject"""
        print(f"  📊 Visualizing data for subject {subject_id:02d}...")
        
        # Select examples from each class
        class_0_indices = np.where(labels == 0)[0][:n_examples//2]
        class_1_indices = np.where(labels == 1)[0][:n_examples//2]
        selected_indices = np.concatenate([class_0_indices, class_1_indices])
        
        fig, axes = plt.subplots(2, n_examples, figsize=(15, 8))
        
        for i, idx in enumerate(selected_indices):
            # Plot EEG data (average across channels)
            axes[0, i].plot(np.mean(eeg_data[idx], axis=0))
            axes[0, i].set_title(f"Trial {idx}, Class {labels[idx]}")
            axes[0, i].set_ylabel("EEG (μV)")
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot EEG topography (simplified)
            axes[1, i].imshow(eeg_data[idx], aspect='auto', cmap='RdBu_r')
            axes[1, i].set_title("EEG Channels")
            axes[1, i].set_ylabel("Channels")
            axes[1, i].set_xlabel("Time")
        
        plt.suptitle(f'THINGS-EEG Subject {subject_id:02d} - Visual Perception Data', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'things_eeg_subject_{subject_id:02d}_visualization.png', dpi=300, bbox_inches='tight')
        print(f"    ✅ Visualization saved as 'things_eeg_subject_{subject_id:02d}_visualization.png'")
        
        plt.close()

def main():
    """Main function to demonstrate THINGS-EEG dataset usage"""
    print("🧠 THINGS-EEG Dataset Loader")
    print("=" * 40)
    
    # Initialize dataset
    dataset = THINGSEEGDataset()
    
    # Show download information
    dataset.download_info()
    
    # Validate dataset
    is_valid, available_subjects = dataset.validate_dataset()
    
    if is_valid and len(available_subjects) > 0:
        # Load and process first available subject
        first_subject = available_subjects[0]
        eeg_data, labels, beh_data = dataset.load_subject_data(first_subject, max_trials=200)
        
        if eeg_data is not None:
            # Extract features
            features = dataset.extract_visual_features(eeg_data)
            
            # Visualize data
            dataset.visualize_subject_data(first_subject, eeg_data, labels)
            
            print(f"\n✅ THINGS-EEG dataset processing completed!")
            print(f"📊 Ready for cross-subject validation with {len(available_subjects)} subjects")
            print(f"📊 Features shape: {features.shape}")
            
            # Save processed data for later use
            np.savez(f'things_eeg_subject_{first_subject:02d}_processed.npz',
                    eeg_data=eeg_data,
                    labels=labels,
                    features=features)
            print(f"💾 Processed data saved as 'things_eeg_subject_{first_subject:02d}_processed.npz'")
        
        else:
            print("❌ Failed to load subject data")
    
    else:
        print("❌ Dataset validation failed")
        print("💡 Please download THINGS-EEG dataset from: https://osf.io/crn2h/")

if __name__ == "__main__":
    main()

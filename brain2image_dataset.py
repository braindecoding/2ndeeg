#!/usr/bin/env python3
# brain2image_dataset.py - Brain2Image Dataset loader and processor

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import mne
from scipy.signal import resample
import h5py
from PIL import Image

class Brain2ImageDataset:
    """Brain2Image Dataset loader and processor"""
    
    def __init__(self, data_path="datasets/brain2image"):
        self.data_path = data_path
        self.subjects = []
        self.image_categories = ['faces', 'objects', 'scenes']
        self.sampling_rate = 1000  # Hz
        self.n_channels = 64
        self.trial_duration = 1.0  # seconds
        
    def download_info(self):
        """Display download information for Brain2Image Dataset"""
        print("📥 Brain2Image Dataset Information")
        print("=" * 50)
        print("🔗 Dataset: Brain2Image - EEG-based Image Reconstruction")
        print("📄 Paper: 'Reconstructing Perceived Images from Brain Activity'")
        print("🌐 Contact: Dataset authors for access")
        print()
        print("📋 Dataset Characteristics:")
        print("  - Multiple subjects (typically 6-10)")
        print("  - 64 EEG channels")
        print("  - 1000 Hz sampling rate")
        print("  - Visual perception paradigm")
        print("  - Image categories: faces, objects, scenes")
        print("  - High-quality EEG data")
        print()
        print("📁 Expected file structure:")
        print("  datasets/brain2image/")
        print("  ├── subject_01/")
        print("  │   ├── eeg_data.mat")
        print("  │   ├── image_labels.mat")
        print("  │   └── images/")
        print("  ├── subject_02/")
        print("  └── ...")
        print()
        print("⚠️ Note: Contact dataset authors for access")
        
    def create_demo_brain2image_data(self):
        """Create demo data that simulates Brain2Image dataset"""
        print("🔧 Creating demo Brain2Image dataset...")
        
        os.makedirs(self.data_path, exist_ok=True)
        
        # Create data for 4 subjects
        n_subjects = 4
        n_trials_per_category = 60
        n_channels = 14  # Match our setup
        n_timepoints = 128  # 1 second at 128 Hz
        
        for subject_id in range(1, n_subjects + 1):
            print(f"  🔧 Creating data for Subject {subject_id}...")
            
            subject_dir = os.path.join(self.data_path, f"subject_{subject_id:02d}")
            os.makedirs(subject_dir, exist_ok=True)
            
            # Set different random seed for each subject
            np.random.seed(42 + subject_id * 200)
            
            all_trials = []
            all_labels = []
            all_images = []
            
            # Subject-specific characteristics
            subject_noise = 0.15 + 0.1 * np.random.random()
            subject_amplitude = 1.2 + 0.4 * np.random.random()
            subject_latency_shift = np.random.uniform(-0.02, 0.02)  # ±20ms shift
            
            for category_id, category in enumerate(['faces', 'objects']):  # 2 categories
                for trial in range(n_trials_per_category):
                    # Create realistic EEG response to visual stimuli
                    base_signal = np.random.randn(n_channels, n_timepoints) * subject_noise
                    
                    # Time axis (1 second)
                    time_axis = np.linspace(0, 1, n_timepoints)
                    
                    if category == 'faces':
                        # Face-specific ERP components
                        # P100 component (visual processing)
                        p100_latency = 0.1 + subject_latency_shift
                        p100_response = subject_amplitude * np.exp(-((time_axis - p100_latency) / 0.02)**2)
                        base_signal[-4:, :] += p100_response * 1.5  # Occipital
                        
                        # N170 component (face-specific)
                        n170_latency = 0.17 + subject_latency_shift
                        n170_response = -subject_amplitude * 1.2 * np.exp(-((time_axis - n170_latency) / 0.025)**2)
                        base_signal[-6:-2, :] += n170_response  # Temporal-occipital
                        
                        # P300 component (recognition)
                        p300_latency = 0.3 + subject_latency_shift
                        p300_response = subject_amplitude * 0.8 * np.exp(-((time_axis - p300_latency) / 0.05)**2)
                        base_signal[4:8, :] += p300_response  # Parietal
                        
                    else:  # objects
                        # Object-specific responses
                        # P100 (general visual)
                        p100_latency = 0.1 + subject_latency_shift
                        p100_response = subject_amplitude * 0.9 * np.exp(-((time_axis - p100_latency) / 0.02)**2)
                        base_signal[-4:, :] += p100_response  # Occipital
                        
                        # N200 component (object processing)
                        n200_latency = 0.2 + subject_latency_shift
                        n200_response = -subject_amplitude * 0.7 * np.exp(-((time_axis - n200_latency) / 0.03)**2)
                        base_signal[-6:-2, :] += n200_response
                        
                        # P400 component (object recognition)
                        p400_latency = 0.4 + subject_latency_shift
                        p400_response = subject_amplitude * 0.6 * np.exp(-((time_axis - p400_latency) / 0.06)**2)
                        base_signal[6:10, :] += p400_response  # Temporal
                    
                    # Add alpha suppression during visual processing
                    alpha_freq = 10 + np.random.uniform(-1, 1)  # Individual alpha frequency
                    alpha_suppression = -0.4 * subject_amplitude * np.sin(2 * np.pi * alpha_freq * time_axis)
                    base_signal[-2:, :] += alpha_suppression  # Occipital alpha
                    
                    # Add gamma activity (visual binding)
                    gamma_freq = 40 + np.random.uniform(-5, 5)
                    gamma_activity = 0.15 * subject_amplitude * np.sin(2 * np.pi * gamma_freq * time_axis)
                    base_signal[-4:, :] += gamma_activity
                    
                    # Create corresponding "image" (simplified representation)
                    if category == 'faces':
                        # Face-like pattern
                        image = np.random.rand(32, 32) * 0.3 + 0.5
                        # Add face-like features
                        image[8:12, 12:20] = 0.8  # Eyes region
                        image[20:24, 14:18] = 0.2  # Nose region
                        image[26:28, 10:22] = 0.7  # Mouth region
                    else:
                        # Object-like pattern
                        image = np.random.rand(32, 32) * 0.6 + 0.2
                        # Add object-like features (more geometric)
                        image[10:22, 10:22] = 0.9  # Central object
                    
                    all_trials.append(base_signal)
                    all_labels.append(category_id)
                    all_images.append(image)
            
            # Convert to numpy arrays
            eeg_data = np.array(all_trials)
            labels = np.array(all_labels)
            images = np.array(all_images)
            
            # Shuffle data
            indices = np.random.permutation(len(eeg_data))
            eeg_data = eeg_data[indices]
            labels = labels[indices]
            images = images[indices]
            
            # Save data
            np.savez(os.path.join(subject_dir, 'brain2image_data.npz'),
                    eeg_data=eeg_data,
                    labels=labels,
                    images=images,
                    subject_characteristics={
                        'noise_level': subject_noise,
                        'amplitude': subject_amplitude,
                        'latency_shift': subject_latency_shift
                    })
            
            print(f"    ✅ Subject {subject_id}: {eeg_data.shape}, {len(np.unique(labels))} categories")
            
        print(f"  ✅ Demo Brain2Image dataset created in {self.data_path}")
        return self.data_path
    
    def load_subject_data(self, subject_id):
        """Load Brain2Image data for a specific subject"""
        print(f"📂 Loading Brain2Image data for subject {subject_id:02d}...")
        
        try:
            subject_dir = os.path.join(self.data_path, f"subject_{subject_id:02d}")
            data_file = os.path.join(subject_dir, 'brain2image_data.npz')
            
            if not os.path.exists(data_file):
                print(f"  ❌ File not found: {data_file}")
                return None, None, None
            
            # Load data
            data = np.load(data_file, allow_pickle=True)
            eeg_data = data['eeg_data']
            labels = data['labels']
            images = data['images']
            characteristics = data['subject_characteristics'].item()
            
            print(f"  ✅ Loaded {len(eeg_data)} trials for subject {subject_id:02d}")
            print(f"  📊 EEG data shape: {eeg_data.shape}")
            print(f"  📊 Images shape: {images.shape}")
            print(f"  📊 Category distribution: {np.bincount(labels)}")
            print(f"  📊 Subject characteristics: {characteristics}")
            
            return eeg_data, labels, images
            
        except Exception as e:
            print(f"  ❌ Error loading Brain2Image data: {str(e)}")
            return None, None, None
    
    def extract_brain2image_features(self, eeg_data):
        """Extract features optimized for image reconstruction tasks"""
        print("  🧩 Extracting Brain2Image features...")
        
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
                    np.percentile(channel_data, 75)
                ])
                
                # Visual ERP components (time windows)
                # P100 window (80-120ms)
                p100_start, p100_end = int(0.08 * 128), int(0.12 * 128)
                p100_mean = np.mean(channel_data[p100_start:p100_end]) if p100_end <= len(channel_data) else 0
                p100_max = np.max(channel_data[p100_start:p100_end]) if p100_end <= len(channel_data) else 0
                
                # N170 window (150-200ms) - face processing
                n170_start, n170_end = int(0.15 * 128), int(0.20 * 128)
                n170_mean = np.mean(channel_data[n170_start:n170_end]) if n170_end <= len(channel_data) else 0
                n170_min = np.min(channel_data[n170_start:n170_end]) if n170_end <= len(channel_data) else 0
                
                # P300 window (250-400ms) - recognition
                p300_start, p300_end = int(0.25 * 128), int(0.40 * 128)
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
                high_gamma_mask = (freqs >= 50) & (freqs <= 80)
                
                # Band powers
                delta_power = np.mean(power_spectrum[delta_mask]) if np.any(delta_mask) else 0
                theta_power = np.mean(power_spectrum[theta_mask]) if np.any(theta_mask) else 0
                alpha_power = np.mean(power_spectrum[alpha_mask]) if np.any(alpha_mask) else 0
                beta_power = np.mean(power_spectrum[beta_mask]) if np.any(beta_mask) else 0
                gamma_power = np.mean(power_spectrum[gamma_mask]) if np.any(gamma_mask) else 0
                high_gamma_power = np.mean(power_spectrum[high_gamma_mask]) if np.any(high_gamma_mask) else 0
                
                trial_features.extend([delta_power, theta_power, alpha_power, beta_power, gamma_power, high_gamma_power])
                
                # Spectral features
                peak_freq = freqs[np.argmax(power_spectrum)] if len(power_spectrum) > 0 else 0
                spectral_centroid = np.sum(freqs * power_spectrum) / (np.sum(power_spectrum) + 1e-10)
                
                trial_features.extend([peak_freq, spectral_centroid])
            
            # Cross-channel features for visual processing
            # Visual cortex regions
            occipital_channels = trial[-4:, :]  # Posterior channels
            temporal_channels = trial[4:8, :]   # Temporal channels
            frontal_channels = trial[:4, :]     # Frontal channels
            parietal_channels = trial[8:12, :]  # Parietal channels
            
            # Regional averages
            occipital_avg = np.mean(occipital_channels, axis=0)
            temporal_avg = np.mean(temporal_channels, axis=0)
            frontal_avg = np.mean(frontal_channels, axis=0)
            parietal_avg = np.mean(parietal_channels, axis=0)
            
            # Inter-regional correlations (important for visual processing)
            occipital_temporal_corr = np.corrcoef(occipital_avg, temporal_avg)[0, 1]
            occipital_frontal_corr = np.corrcoef(occipital_avg, frontal_avg)[0, 1]
            temporal_parietal_corr = np.corrcoef(temporal_avg, parietal_avg)[0, 1]
            
            trial_features.extend([occipital_temporal_corr, occipital_frontal_corr, temporal_parietal_corr])
            
            # Global measures
            # Global field power
            gfp = np.std(trial, axis=0)
            gfp_mean = np.mean(gfp)
            gfp_max = np.max(gfp)
            gfp_std = np.std(gfp)
            
            # Microstate-like features (simplified)
            trial_std = np.std(trial, axis=1)  # Variability per channel
            spatial_complexity = np.std(trial_std)
            
            trial_features.extend([gfp_mean, gfp_max, gfp_std, spatial_complexity])
            
            features.append(trial_features)
        
        features = np.array(features, dtype=np.float64)
        
        # Handle NaN values
        features = np.nan_to_num(features)
        
        print(f"    ✅ Brain2Image features extracted: {features.shape}")
        return features
    
    def visualize_subject_data(self, subject_id, eeg_data, labels, images, n_examples=4):
        """Visualize EEG data and corresponding images for a subject"""
        print(f"  📊 Visualizing data for subject {subject_id:02d}...")
        
        # Select examples from each category
        category_0_indices = np.where(labels == 0)[0][:n_examples//2]
        category_1_indices = np.where(labels == 1)[0][:n_examples//2]
        selected_indices = np.concatenate([category_0_indices, category_1_indices])
        
        fig, axes = plt.subplots(3, n_examples, figsize=(15, 10))
        
        for i, idx in enumerate(selected_indices):
            # Plot EEG data (average across channels)
            axes[0, i].plot(np.mean(eeg_data[idx], axis=0))
            axes[0, i].set_title(f"Trial {idx}, Category {labels[idx]}")
            axes[0, i].set_ylabel("EEG (μV)")
            
            # Plot EEG topography (simplified)
            axes[1, i].imshow(eeg_data[idx], aspect='auto', cmap='RdBu_r')
            axes[1, i].set_title("EEG Channels")
            axes[1, i].set_ylabel("Channels")
            
            # Plot corresponding image
            axes[2, i].imshow(images[idx], cmap='gray')
            axes[2, i].set_title("Image")
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'brain2image_subject_{subject_id:02d}_visualization.png', dpi=300, bbox_inches='tight')
        print(f"    ✅ Visualization saved as 'brain2image_subject_{subject_id:02d}_visualization.png'")
        
        plt.close()

def main():
    """Main function to demonstrate Brain2Image dataset usage"""
    print("🧠 Brain2Image Dataset Demo")
    print("=" * 40)
    
    # Initialize dataset
    dataset = Brain2ImageDataset()
    
    # Show download information
    dataset.download_info()
    
    # Create demo data
    data_path = dataset.create_demo_brain2image_data()
    
    # Load and visualize data for first subject
    eeg_data, labels, images = dataset.load_subject_data(1)
    
    if eeg_data is not None:
        # Extract features
        features = dataset.extract_brain2image_features(eeg_data)
        
        # Visualize data
        dataset.visualize_subject_data(1, eeg_data, labels, images)
        
        print(f"\n✅ Brain2Image dataset demo completed!")
        print(f"📊 Ready for cross-subject validation")
    
    else:
        print("❌ Failed to load demo data")

if __name__ == "__main__":
    main()

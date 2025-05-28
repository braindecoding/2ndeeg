#!/usr/bin/env python3
# visual_eeg_dataset_loader.py - Loader for visual EEG dataset (.fif files)

import numpy as np
import os
import mne
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

class VisualEEGDataset:
    """Visual EEG Dataset loader for .fif files"""

    def __init__(self, data_path="datasets"):
        self.data_path = Path(data_path)
        self.target_sampling_rate = 128  # Downsample to match our models
        self.target_channels = 14  # Downsample channels to match our models

    def setup_dataset_directory(self):
        """Create dataset directory and move files"""
        print("📁 Setting up dataset directory...")

        # Create directory
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Check if files are in current directory
        current_files = [
            "subj01_session1_eeg.fif",
            "subj01_session2_eeg.fif"
        ]

        moved_files = []
        for filename in current_files:
            if os.path.exists(filename):
                # Move to dataset directory
                new_path = self.data_path / filename
                if not new_path.exists():
                    os.rename(filename, new_path)
                    print(f"  ✅ Moved {filename} to {new_path}")
                else:
                    print(f"  ✅ {filename} already in dataset directory")
                moved_files.append(str(new_path))
            else:
                # Check if already in dataset directory
                dataset_path = self.data_path / filename
                if dataset_path.exists():
                    print(f"  ✅ Found {filename} in dataset directory")
                    moved_files.append(str(dataset_path))
                else:
                    print(f"  ❌ {filename} not found")

        return moved_files

    def validate_dataset(self):
        """Validate downloaded dataset"""
        print("🔍 Validating visual EEG dataset...")

        if not self.data_path.exists():
            print(f"❌ Dataset directory not found: {self.data_path}")
            return False, []

        # Look for .fif files (including files with "copy" in name)
        fif_files = []
        for pattern in ["*.fif", "*copy*"]:
            fif_files.extend(self.data_path.glob(pattern))

        # Filter to only .fif files
        fif_files = [f for f in fif_files if f.suffix == '.fif' or 'eeg.fif' in f.name]

        if len(fif_files) == 0:
            print("❌ No .fif files found in dataset directory")
            print("💡 Please ensure .fif files are in the dataset directory")
            return False, []

        print(f"✅ Found {len(fif_files)} .fif files:")
        for fif_file in fif_files:
            file_size = fif_file.stat().st_size / (1024**2)  # MB
            print(f"  📄 {fif_file.name} ({file_size:.1f} MB)")

        return True, fif_files

    def load_fif_file(self, fif_file):
        """Load a single .fif file"""
        print(f"📂 Loading {fif_file.name}...")

        try:
            # Load raw data
            raw = mne.io.read_raw_fif(str(fif_file), preload=True, verbose=False)

            print(f"  📊 Sampling rate: {raw.info['sfreq']} Hz")
            print(f"  📊 Channels: {len(raw.ch_names)}")
            print(f"  📊 Duration: {raw.times[-1]:.1f} seconds")
            print(f"  📊 Channel names: {raw.ch_names[:10]}...")  # First 10 channels

            return raw

        except Exception as e:
            print(f"  ❌ Error loading {fif_file}: {str(e)}")
            return None

    def extract_trials_from_raw(self, raw, trial_duration=1.0, overlap=0.5):
        """Extract trials from continuous raw data"""
        print(f"  🔧 Extracting trials (duration={trial_duration}s, overlap={overlap})...")

        try:
            # Get data
            data = raw.get_data()  # Shape: [channels, timepoints]
            sfreq = raw.info['sfreq']

            # Calculate trial parameters
            trial_samples = int(trial_duration * sfreq)
            step_samples = int(trial_samples * (1 - overlap))

            # Extract overlapping trials
            trials = []
            n_trials = (data.shape[1] - trial_samples) // step_samples + 1

            for i in range(n_trials):
                start_idx = i * step_samples
                end_idx = start_idx + trial_samples

                if end_idx <= data.shape[1]:
                    trial = data[:, start_idx:end_idx]
                    trials.append(trial)

            trials = np.array(trials)  # Shape: [n_trials, n_channels, n_timepoints]

            print(f"    ✅ Extracted {len(trials)} trials")
            print(f"    📊 Trial shape: {trials.shape}")

            return trials

        except Exception as e:
            print(f"    ❌ Error extracting trials: {str(e)}")
            return None

    def preprocess_trials(self, trials):
        """Preprocess trials to match our model format"""
        print("  🔧 Preprocessing trials...")

        try:
            # Downsample channels if necessary
            if trials.shape[1] > self.target_channels:
                # Select subset of channels (distributed across scalp)
                channel_indices = np.linspace(0, trials.shape[1]-1, self.target_channels, dtype=int)
                trials = trials[:, channel_indices, :]
                print(f"    📊 Downsampled channels: {trials.shape[1]}")

            # Downsample time points to 128 (to match our models)
            if trials.shape[2] != 128:
                trials_resampled = []
                for trial in trials:
                    trial_resampled = resample(trial, 128, axis=1)
                    trials_resampled.append(trial_resampled)
                trials = np.array(trials_resampled)
                print(f"    📊 Resampled timepoints: {trials.shape[2]}")

            print(f"    ✅ Final shape: {trials.shape}")
            return trials

        except Exception as e:
            print(f"    ❌ Error preprocessing: {str(e)}")
            return None

    def create_labels_for_sessions(self, session1_trials, session2_trials):
        """Create binary labels for two sessions"""
        print("  🏷️ Creating labels for sessions...")

        # Session 1 = class 0, Session 2 = class 1
        labels1 = np.zeros(len(session1_trials), dtype=int)
        labels2 = np.ones(len(session2_trials), dtype=int)

        all_labels = np.concatenate([labels1, labels2])

        print(f"    📊 Session 1 trials: {len(session1_trials)} (label 0)")
        print(f"    📊 Session 2 trials: {len(session2_trials)} (label 1)")
        print(f"    📊 Total labels: {len(all_labels)}")
        print(f"    📊 Label distribution: {np.bincount(all_labels)}")

        return all_labels

    def load_all_data(self):
        """Load all available data"""
        print("🚀 Loading Visual EEG Dataset")
        print("=" * 50)

        # Setup and validate
        moved_files = self.setup_dataset_directory()
        is_valid, fif_files = self.validate_dataset()

        if not is_valid:
            return None, None

        all_trials = []
        session_info = []

        # Load each .fif file
        for fif_file in fif_files:
            raw = self.load_fif_file(fif_file)

            if raw is not None:
                # Extract trials
                trials = self.extract_trials_from_raw(raw, trial_duration=1.0, overlap=0.5)

                if trials is not None:
                    # Preprocess
                    trials_processed = self.preprocess_trials(trials)

                    if trials_processed is not None:
                        all_trials.append(trials_processed)
                        session_info.append({
                            'filename': fif_file.name,
                            'n_trials': len(trials_processed),
                            'session_id': 1 if 'session1' in fif_file.name else 2
                        })

        if len(all_trials) == 0:
            print("❌ No trials extracted from any file")
            return None, None

        # Combine all trials
        combined_trials = np.concatenate(all_trials, axis=0)

        # Create labels based on sessions
        if len(all_trials) == 2:
            # Assume first file is session 1, second is session 2
            labels = self.create_labels_for_sessions(all_trials[0], all_trials[1])
        else:
            # Create artificial labels if only one session
            print("  ⚠️ Only one session found, creating artificial binary labels")
            n_trials = len(combined_trials)
            labels = np.array([i % 2 for i in range(n_trials)])

        print(f"\n✅ Dataset loaded successfully!")
        print(f"📊 Total trials: {len(combined_trials)}")
        print(f"📊 Final data shape: {combined_trials.shape}")
        print(f"📊 Labels shape: {labels.shape}")
        print(f"📊 Session info: {session_info}")

        return combined_trials, labels

    def extract_visual_features(self, eeg_data):
        """Extract features optimized for visual perception tasks"""
        print("🧩 Extracting visual perception features...")

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

                # Visual ERP components (assuming 128 timepoints = 1 second)
                # P100 window (80-120ms)
                p100_start, p100_end = int(0.08 * 128), int(0.12 * 128)
                p100_mean = np.mean(channel_data[p100_start:p100_end]) if p100_end <= len(channel_data) else 0
                p100_max = np.max(channel_data[p100_start:p100_end]) if p100_end <= len(channel_data) else 0

                # N170 window (150-200ms) - visual processing
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

            # Global field power
            gfp = np.std(trial, axis=0)
            gfp_mean = np.mean(gfp)
            gfp_max = np.max(gfp)
            gfp_std = np.std(gfp)

            trial_features.extend([gfp_mean, gfp_max, gfp_std])

            features.append(trial_features)

        features = np.array(features, dtype=np.float64)
        features = np.nan_to_num(features)

        print(f"  ✅ Visual perception features extracted: {features.shape}")
        return features

    def visualize_data(self, eeg_data, labels, n_examples=4):
        """Visualize EEG data"""
        print("📊 Creating data visualization...")

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

        plt.suptitle('Visual EEG Dataset - Real Data', fontsize=14)
        plt.tight_layout()
        plt.savefig('visual_eeg_dataset_visualization.png', dpi=300, bbox_inches='tight')
        print("  ✅ Visualization saved as 'visual_eeg_dataset_visualization.png'")

        plt.close()

def main():
    """Main function to test the loader"""
    print("🧠 Visual EEG Dataset Loader")
    print("=" * 40)

    # Initialize dataset
    dataset = VisualEEGDataset()

    # Load all data
    eeg_data, labels = dataset.load_all_data()

    if eeg_data is not None:
        # Extract features
        features = dataset.extract_visual_features(eeg_data)

        # Visualize data
        dataset.visualize_data(eeg_data, labels)

        # Save processed data
        np.savez('visual_eeg_processed_data.npz',
                eeg_data=eeg_data,
                labels=labels,
                features=features)

        print(f"\n✅ Visual EEG dataset processing completed!")
        print(f"📊 EEG data: {eeg_data.shape}")
        print(f"📊 Labels: {labels.shape}")
        print(f"📊 Features: {features.shape}")
        print(f"💾 Data saved as 'visual_eeg_processed_data.npz'")
        print(f"🚀 Ready for cross-subject validation!")

    else:
        print("❌ Failed to load visual EEG dataset")

if __name__ == "__main__":
    main()

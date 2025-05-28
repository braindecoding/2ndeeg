#!/usr/bin/env python3
# download_physionet_eeg.py - Download PhysioNet EEG Motor Movement/Imagery Dataset

import os
import urllib.request
import urllib.error
from pathlib import Path

def create_physionet_directory():
    """Create PhysioNet dataset directory"""
    print("📁 Creating PhysioNet dataset directory...")
    
    dataset_dir = Path("datasets/physionet_eeg")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory: {dataset_dir}")
    return dataset_dir

def download_physionet_subject(subject_id, dataset_dir):
    """Download PhysioNet data for a specific subject"""
    print(f"📥 Downloading PhysioNet subject {subject_id:03d}...")
    
    base_url = "https://physionet.org/files/eegmmidb/1.0.0/"
    subject_dir = dataset_dir / f"S{subject_id:03d}"
    subject_dir.mkdir(exist_ok=True)
    
    # Download specific runs for motor imagery
    runs_to_download = [3, 7, 11, 15]  # Motor imagery tasks
    downloaded_files = []
    
    for run_id in runs_to_download:
        filename = f"S{subject_id:03d}R{run_id:02d}.edf"
        url = f"{base_url}S{subject_id:03d}/{filename}"
        filepath = subject_dir / filename
        
        if filepath.exists():
            print(f"  ✅ {filename} already exists")
            downloaded_files.append(str(filepath))
            continue
        
        try:
            print(f"  📥 Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            
            # Check file size
            file_size = filepath.stat().st_size / (1024**2)  # MB
            print(f"    ✅ Downloaded {filename} ({file_size:.1f} MB)")
            downloaded_files.append(str(filepath))
            
        except urllib.error.URLError as e:
            print(f"    ❌ Failed to download {filename}: {str(e)}")
        except Exception as e:
            print(f"    ❌ Error downloading {filename}: {str(e)}")
    
    return downloaded_files

def download_multiple_subjects(n_subjects=3):
    """Download multiple subjects from PhysioNet"""
    print(f"🚀 Downloading PhysioNet EEG data for {n_subjects} subjects...")
    print("📊 Dataset: EEG Motor Movement/Imagery Dataset")
    print("🔗 Source: https://physionet.org/content/eegmmidb/1.0.0/")
    print()
    
    dataset_dir = create_physionet_directory()
    
    all_downloaded_files = []
    successful_subjects = []
    
    for subject_id in range(1, n_subjects + 1):
        print(f"\n📂 Processing Subject {subject_id:03d}")
        print("-" * 40)
        
        downloaded_files = download_physionet_subject(subject_id, dataset_dir)
        
        if len(downloaded_files) > 0:
            all_downloaded_files.extend(downloaded_files)
            successful_subjects.append(subject_id)
            print(f"  ✅ Subject {subject_id:03d}: {len(downloaded_files)} files downloaded")
        else:
            print(f"  ❌ Subject {subject_id:03d}: No files downloaded")
    
    print(f"\n📊 Download Summary:")
    print(f"  Subjects attempted: {n_subjects}")
    print(f"  Subjects successful: {len(successful_subjects)}")
    print(f"  Total files downloaded: {len(all_downloaded_files)}")
    print(f"  Successful subjects: {successful_subjects}")
    
    return successful_subjects, all_downloaded_files

def create_physionet_loader():
    """Create loader script for PhysioNet data"""
    print("\n🔧 Creating PhysioNet data loader...")
    
    loader_code = '''#!/usr/bin/env python3
# physionet_eeg_loader.py - Load and process PhysioNet EEG data

import numpy as np
import mne
from pathlib import Path
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler

class PhysioNetEEGDataset:
    """PhysioNet EEG Dataset loader and processor"""
    
    def __init__(self, data_path="datasets/physionet_eeg"):
        self.data_path = Path(data_path)
        self.sampling_rate = 160  # Hz
        self.target_sampling_rate = 128  # Downsample to match our models
        self.target_channels = 14  # Downsample channels to match our models
    
    def load_subject_data(self, subject_id, max_trials=200):
        """Load PhysioNet data for a specific subject"""
        print(f"📂 Loading PhysioNet subject {subject_id:03d}...")
        
        subject_dir = self.data_path / f"S{subject_id:03d}"
        
        if not subject_dir.exists():
            print(f"  ❌ Subject directory not found: {subject_dir}")
            return None, None
        
        # Load motor imagery runs
        runs = [3, 7, 11, 15]  # Left hand, right hand, feet, tongue imagery
        all_trials = []
        all_labels = []
        
        for run_id in runs:
            edf_file = subject_dir / f"S{subject_id:03d}R{run_id:02d}.edf"
            
            if not edf_file.exists():
                print(f"    ⚠️ File not found: {edf_file}")
                continue
            
            try:
                # Load EDF file
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
                
                # Get events
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                
                # Extract epochs around events
                if len(events) > 0:
                    epochs = mne.Epochs(raw, events, event_id=None, tmin=-0.1, tmax=0.5,
                                      baseline=(None, 0), preload=True, verbose=False)
                    
                    epochs_data = epochs.get_data()  # [n_epochs, n_channels, n_times]
                    
                    # Downsample channels
                    if epochs_data.shape[1] > self.target_channels:
                        channel_indices = np.linspace(0, epochs_data.shape[1]-1, 
                                                    self.target_channels, dtype=int)
                        epochs_data = epochs_data[:, channel_indices, :]
                    
                    # Downsample time points
                    if epochs_data.shape[2] != 128:
                        epochs_data_resampled = []
                        for epoch in epochs_data:
                            epoch_resampled = resample(epoch, 128, axis=1)
                            epochs_data_resampled.append(epoch_resampled)
                        epochs_data = np.array(epochs_data_resampled)
                    
                    # Create labels (binary classification)
                    # Run 3,11 = class 0 (left hand, feet)
                    # Run 7,15 = class 1 (right hand, tongue)
                    label = 0 if run_id in [3, 11] else 1
                    labels = np.full(len(epochs_data), label)
                    
                    all_trials.extend(epochs_data)
                    all_labels.extend(labels)
                    
                    print(f"    ✅ Run {run_id:02d}: {len(epochs_data)} trials, label {label}")
                
            except Exception as e:
                print(f"    ❌ Error loading {edf_file}: {str(e)}")
        
        if len(all_trials) == 0:
            print(f"  ❌ No trials extracted for subject {subject_id}")
            return None, None
        
        # Convert to numpy arrays
        eeg_data = np.array(all_trials)
        labels = np.array(all_labels)
        
        # Limit trials if requested
        if len(eeg_data) > max_trials:
            indices = np.random.choice(len(eeg_data), max_trials, replace=False)
            eeg_data = eeg_data[indices]
            labels = labels[indices]
        
        print(f"  ✅ Loaded {len(eeg_data)} trials")
        print(f"  📊 Data shape: {eeg_data.shape}")
        print(f"  📊 Label distribution: {np.bincount(labels)}")
        
        return eeg_data, labels
    
    def extract_features(self, eeg_data):
        """Extract features from EEG data"""
        print("  🧩 Extracting features...")
        
        features = []
        
        for trial in eeg_data:
            trial_features = []
            
            # Statistical features for each channel
            for channel in range(trial.shape[0]):
                channel_data = trial[channel]
                
                trial_features.extend([
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.var(channel_data),
                    np.max(channel_data),
                    np.min(channel_data),
                    np.median(channel_data)
                ])
                
                # Frequency domain features
                fft = np.fft.fft(channel_data)
                power_spectrum = np.abs(fft[:len(fft)//2])**2
                freqs = np.fft.fftfreq(len(channel_data), 1/128)[:len(fft)//2]
                
                # Band powers
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                beta_mask = (freqs >= 13) & (freqs <= 30)
                
                alpha_power = np.mean(power_spectrum[alpha_mask]) if np.any(alpha_mask) else 0
                beta_power = np.mean(power_spectrum[beta_mask]) if np.any(beta_mask) else 0
                
                trial_features.extend([alpha_power, beta_power])
            
            features.append(trial_features)
        
        features = np.array(features, dtype=np.float64)
        features = np.nan_to_num(features)
        
        print(f"    ✅ Features extracted: {features.shape}")
        return features

def main():
    """Test PhysioNet loader"""
    dataset = PhysioNetEEGDataset()
    
    # Try to load first subject
    eeg_data, labels = dataset.load_subject_data(1)
    
    if eeg_data is not None:
        features = dataset.extract_features(eeg_data)
        print(f"\\n✅ PhysioNet data loaded successfully!")
        print(f"📊 EEG data: {eeg_data.shape}")
        print(f"📊 Features: {features.shape}")
    else:
        print(f"\\n❌ Failed to load PhysioNet data")

if __name__ == "__main__":
    main()
'''
    
    with open('physionet_eeg_loader.py', 'w') as f:
        f.write(loader_code)
    
    print("✅ Created: physionet_eeg_loader.py")

def create_physionet_validation():
    """Create validation script for PhysioNet data"""
    print("🔧 Creating PhysioNet validation script...")
    
    validation_code = '''#!/usr/bin/env python3
# validate_physionet.py - Validate PhysioNet dataset

from pathlib import Path

def validate_physionet():
    """Validate PhysioNet dataset"""
    print("🔍 Validating PhysioNet dataset...")
    
    dataset_dir = Path("datasets/physionet_eeg")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False, []
    
    subjects_found = []
    
    for i in range(1, 11):  # Check first 10 subjects
        subject_dir = dataset_dir / f"S{i:03d}"
        
        if subject_dir.exists():
            edf_files = list(subject_dir.glob("*.edf"))
            if len(edf_files) > 0:
                subjects_found.append(i)
                print(f"  ✅ Subject {i:03d}: {len(edf_files)} files")
    
    print(f"\\n📊 Summary:")
    print(f"  Subjects found: {len(subjects_found)}")
    print(f"  Subject IDs: {subjects_found}")
    
    if len(subjects_found) >= 2:
        print(f"  ✅ Sufficient for cross-subject validation")
        return True, subjects_found
    else:
        print(f"  ❌ Need at least 2 subjects")
        return False, subjects_found

if __name__ == "__main__":
    validate_physionet()
'''
    
    with open('validate_physionet.py', 'w') as f:
        f.write(validation_code)
    
    print("✅ Created: validate_physionet.py")

def main():
    """Main function"""
    print("🚀 PhysioNet EEG Dataset Downloader")
    print("=" * 50)
    print("📊 Dataset: EEG Motor Movement/Imagery Dataset")
    print("🔗 Source: https://physionet.org/content/eegmmidb/1.0.0/")
    print("📋 Features: 109 subjects, 64 channels, motor imagery tasks")
    print()
    
    # Download 3 subjects
    successful_subjects, downloaded_files = download_multiple_subjects(3)
    
    if len(successful_subjects) >= 2:
        print(f"\n🎉 Download successful!")
        print(f"✅ {len(successful_subjects)} subjects downloaded")
        
        # Create helper scripts
        create_physionet_loader()
        create_physionet_validation()
        
        print(f"\n🚀 Next steps:")
        print(f"1. Run: python3 validate_physionet.py")
        print(f"2. Run: python3 physionet_eeg_loader.py")
        print(f"3. Use PhysioNet data for cross-subject validation")
        
    else:
        print(f"\n❌ Download failed or insufficient subjects")
        print(f"💡 Check internet connection and try again")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# setup_alternative_datasets.py - Setup alternative datasets for cross-subject validation

import os
import urllib.request
import zipfile
import tarfile
import numpy as np
from scipy.io import loadmat
import mne

def download_bci_competition_iv_2a():
    """Download BCI Competition IV Dataset 2a"""
    print("📥 Downloading BCI Competition IV Dataset 2a...")
    
    base_url = "http://www.bbci.de/competition/iv/download/"
    data_dir = "datasets/bci_competition_iv_2a"
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Files to download (training data for subjects 1-9)
    files_to_download = [
        "BCICIV_2a_gdf.zip"  # Contains all subjects
    ]
    
    for filename in files_to_download:
        url = base_url + filename
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  📥 Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✅ Downloaded {filename}")
                
                # Extract if it's a zip file
                if filename.endswith('.zip'):
                    with zipfile.ZipFile(filepath, 'r') as zip_ref:
                        zip_ref.extractall(data_dir)
                    print(f"  📂 Extracted {filename}")
                    
            except Exception as e:
                print(f"  ❌ Failed to download {filename}: {str(e)}")
        else:
            print(f"  ✅ {filename} already exists")
    
    return data_dir

def download_physionet_eeg():
    """Download PhysioNet EEG Motor Movement/Imagery Dataset"""
    print("📥 Downloading PhysioNet EEG Dataset...")
    
    base_url = "https://physionet.org/files/eegmmidb/1.0.0/"
    data_dir = "datasets/physionet_eeg"
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Download data for first 10 subjects (for testing)
    subjects_to_download = range(1, 11)  # Subjects S001-S010
    tasks_to_download = [1, 2, 3, 4]  # Baseline and motor tasks
    
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
                except Exception as e:
                    print(f"  ❌ Failed to download {filename}: {str(e)}")
            else:
                print(f"  ✅ {filename} already exists")
    
    return data_dir

def create_synthetic_alternative_dataset():
    """Create synthetic alternative dataset for testing"""
    print("🔧 Creating synthetic alternative dataset...")
    
    data_dir = "datasets/synthetic_alternative"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create synthetic data for 5 subjects
    n_subjects = 5
    n_trials_per_class = 50
    n_channels = 14
    n_timepoints = 128
    n_classes = 2
    
    for subject_id in range(1, n_subjects + 1):
        print(f"  🔧 Creating data for Subject {subject_id}...")
        
        # Generate synthetic EEG data
        np.random.seed(42 + subject_id)  # Different seed for each subject
        
        all_trials = []
        all_labels = []
        
        for class_id in range(n_classes):
            for trial in range(n_trials_per_class):
                # Create synthetic EEG with different patterns for each class
                if class_id == 0:
                    # Class 0: More activity in frontal channels
                    base_signal = np.random.randn(n_channels, n_timepoints) * 0.5
                    base_signal[:4, :] += np.sin(np.linspace(0, 4*np.pi, n_timepoints)) * 2
                else:
                    # Class 1: More activity in posterior channels
                    base_signal = np.random.randn(n_channels, n_timepoints) * 0.5
                    base_signal[-4:, :] += np.cos(np.linspace(0, 6*np.pi, n_timepoints)) * 2
                
                # Add subject-specific characteristics
                subject_factor = 0.8 + 0.4 * (subject_id / n_subjects)
                base_signal *= subject_factor
                
                all_trials.append(base_signal)
                all_labels.append(class_id)
        
        # Convert to numpy arrays
        eeg_data = np.array(all_trials)
        labels = np.array(all_labels)
        
        # Shuffle data
        indices = np.random.permutation(len(eeg_data))
        eeg_data = eeg_data[indices]
        labels = labels[indices]
        
        # Save data
        subject_file = os.path.join(data_dir, f"subject_{subject_id:02d}.npz")
        np.savez(subject_file, eeg_data=eeg_data, labels=labels)
        
        print(f"  ✅ Subject {subject_id} data saved: {eeg_data.shape}")
    
    print(f"  ✅ Synthetic dataset created in {data_dir}")
    return data_dir

def load_synthetic_alternative_data(data_path, subject_id):
    """Load synthetic alternative dataset"""
    print(f"📂 Loading synthetic data for subject {subject_id}...")
    
    try:
        subject_file = os.path.join(data_path, f"subject_{subject_id:02d}.npz")
        
        if not os.path.exists(subject_file):
            print(f"  ❌ File not found: {subject_file}")
            return None, None
        
        # Load data
        data = np.load(subject_file)
        eeg_data = data['eeg_data']
        labels = data['labels']
        
        print(f"  ✅ Loaded {len(eeg_data)} trials for subject {subject_id}")
        print(f"  📊 Data shape: {eeg_data.shape}")
        print(f"  📊 Class distribution: {np.bincount(labels)}")
        
        return eeg_data, labels
        
    except Exception as e:
        print(f"  ❌ Error loading synthetic data: {str(e)}")
        return None, None

def validate_dataset(data_path, dataset_type):
    """Validate that dataset is properly downloaded and formatted"""
    print(f"🔍 Validating {dataset_type} dataset...")
    
    if dataset_type == 'bci_competition':
        # Check for .gdf or .mat files
        expected_files = [f"A0{i}T.gdf" for i in range(1, 10)]
        
    elif dataset_type == 'physionet':
        # Check for .edf files
        expected_files = []
        for subject_id in range(1, 11):
            for task_id in range(1, 5):
                expected_files.append(f"S{subject_id:03d}/S{subject_id:03d}R{task_id:02d}.edf")
                
    elif dataset_type == 'synthetic':
        # Check for .npz files
        expected_files = [f"subject_{i:02d}.npz" for i in range(1, 6)]
    
    else:
        print(f"  ❌ Unknown dataset type: {dataset_type}")
        return False
    
    missing_files = []
    for filename in expected_files[:5]:  # Check first 5 files
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if len(missing_files) == 0:
        print(f"  ✅ {dataset_type} dataset validation passed")
        return True
    else:
        print(f"  ⚠️ Missing files in {dataset_type} dataset:")
        for filename in missing_files:
            print(f"    - {filename}")
        return False

def setup_all_datasets():
    """Setup all alternative datasets"""
    print("🚀 Setting up Alternative Datasets for Cross-Subject Validation")
    print("=" * 60)
    
    datasets_info = {}
    
    # 1. Create synthetic dataset (always available)
    print("\n1. Setting up Synthetic Alternative Dataset")
    print("-" * 40)
    synthetic_path = create_synthetic_alternative_dataset()
    if validate_dataset(synthetic_path, 'synthetic'):
        datasets_info['synthetic'] = synthetic_path
    
    # 2. Try to download BCI Competition IV Dataset 2a
    print("\n2. Setting up BCI Competition IV Dataset 2a")
    print("-" * 40)
    try:
        bci_path = download_bci_competition_iv_2a()
        if validate_dataset(bci_path, 'bci_competition'):
            datasets_info['bci_competition'] = bci_path
        else:
            print("  ⚠️ BCI Competition dataset validation failed")
    except Exception as e:
        print(f"  ❌ Failed to setup BCI Competition dataset: {str(e)}")
    
    # 3. Try to download PhysioNet EEG Dataset
    print("\n3. Setting up PhysioNet EEG Dataset")
    print("-" * 40)
    try:
        physionet_path = download_physionet_eeg()
        if validate_dataset(physionet_path, 'physionet'):
            datasets_info['physionet'] = physionet_path
        else:
            print("  ⚠️ PhysioNet dataset validation failed")
    except Exception as e:
        print(f"  ❌ Failed to setup PhysioNet dataset: {str(e)}")
    
    # Summary
    print(f"\n📊 Dataset Setup Summary:")
    print("-" * 30)
    for dataset_name, dataset_path in datasets_info.items():
        print(f"  ✅ {dataset_name}: {dataset_path}")
    
    if len(datasets_info) == 0:
        print("  ❌ No datasets were successfully setup")
    else:
        print(f"  🎉 {len(datasets_info)} dataset(s) ready for cross-subject validation")
    
    return datasets_info

def test_cross_subject_validation():
    """Test cross-subject validation with available datasets"""
    print("\n🧪 Testing Cross-Subject Validation")
    print("=" * 40)
    
    # Setup datasets
    datasets_info = setup_all_datasets()
    
    if 'synthetic' in datasets_info:
        print("\n🔬 Running cross-subject validation on synthetic dataset...")
        
        # Import our cross-subject validation module
        from cross_subject_validation import cross_subject_validation_pipeline
        
        # Test with synthetic dataset
        try:
            results = cross_subject_validation_pipeline(
                dataset_type='synthetic',
                data_path=datasets_info['synthetic']
            )
            
            if results:
                print("  ✅ Cross-subject validation test passed!")
            else:
                print("  ❌ Cross-subject validation test failed!")
                
        except Exception as e:
            print(f"  ❌ Error in cross-subject validation: {str(e)}")
    
    else:
        print("  ❌ No datasets available for testing")

def main():
    """Main function"""
    print("🚀 Alternative Datasets Setup for EEG Cross-Subject Validation")
    print("=" * 60)
    
    # Setup all datasets
    datasets_info = setup_all_datasets()
    
    # Test cross-subject validation
    test_cross_subject_validation()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Run cross_subject_validation.py with your preferred dataset")
    print("2. Compare results across different subjects")
    print("3. Analyze model generalization performance")

if __name__ == "__main__":
    main()

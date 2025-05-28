#!/usr/bin/env python3
# real_dataset_download_guide.py - Guide for downloading real EEG visual perception datasets

import os
import urllib.request
import zipfile

def download_things_eeg_info():
    """Provide information and download links for THINGS-EEG dataset"""
    print("📥 THINGS-EEG Dataset - Real Visual Perception EEG Data")
    print("=" * 60)
    print()
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
    print("  - Visual perception paradigm (subjects see real images)")
    print("  - High-quality preprocessed data")
    print()
    print("📁 Dataset Structure:")
    print("  THINGS-EEG/")
    print("  ├── sub-01/")
    print("  │   ├── eeg/")
    print("  │   │   └── sub-01_task-rsvp_eeg.set")
    print("  │   └── beh/")
    print("  │       └── sub-01_task-rsvp_beh.tsv")
    print("  ├── sub-02/")
    print("  └── ...")
    print()
    print("💾 Download Instructions:")
    print("  1. Go to: https://osf.io/crn2h/")
    print("  2. Click 'Download as zip' (Warning: ~50GB)")
    print("  3. Extract to 'datasets/things_eeg/'")
    print("  4. Run our validation script")
    print()
    print("⚡ Quick Start (Subset):")
    print("  - Download only 2-3 subjects for testing")
    print("  - Each subject is ~5GB")
    print("  - Sufficient for cross-subject validation")

def download_eeg_imagenet_info():
    """Provide information for EEG-ImageNet dataset"""
    print("\n📥 EEG-ImageNet Dataset - ImageNet Visual Perception")
    print("=" * 60)
    print()
    print("🔗 Access Information:")
    print("  Paper: 'Learning to Reconstruct Perceived Images from Brain Activity'")
    print("  Contact: Dataset authors for access")
    print("  Alternative: Search for 'EEG ImageNet visual perception dataset'")
    print()
    print("📋 Dataset Characteristics:")
    print("  - 6 subjects viewing ImageNet images")
    print("  - 40,000 images from 1,000 categories")
    print("  - 128 EEG channels")
    print("  - 1000 Hz sampling rate")
    print("  - Visual perception paradigm")

def download_physionet_visual_info():
    """Information about PhysioNet visual datasets"""
    print("\n📥 PhysioNet Visual EEG Datasets")
    print("=" * 60)
    print()
    print("🔗 Available Datasets:")
    print("  1. EEG Motor Movement/Imagery Dataset")
    print("     URL: https://physionet.org/content/eegmmidb/1.0.0/")
    print("     Note: Motor imagery, not visual perception")
    print()
    print("  2. EEG During Mental Arithmetic Tasks")
    print("     URL: https://physionet.org/content/eegmat/1.0.0/")
    print("     Note: Cognitive tasks, not visual")
    print()
    print("⚠️ Note: PhysioNet has limited visual perception datasets")

def create_dataset_validation_script():
    """Create script to validate downloaded datasets"""
    print("\n🔧 Creating dataset validation script...")
    
    validation_script = '''#!/usr/bin/env python3
# validate_real_datasets.py - Validate downloaded real EEG datasets

import os
import mne
import numpy as np

def validate_things_eeg(data_path="datasets/things_eeg"):
    """Validate THINGS-EEG dataset"""
    print("🔍 Validating THINGS-EEG dataset...")
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset path not found: {data_path}")
        return False
    
    # Check for subject directories
    subjects_found = []
    for i in range(1, 11):  # 10 subjects
        subject_dir = os.path.join(data_path, f"sub-{i:02d}")
        if os.path.exists(subject_dir):
            subjects_found.append(i)
            
            # Check for EEG file
            eeg_file = os.path.join(subject_dir, "eeg", f"sub-{i:02d}_task-rsvp_eeg.set")
            if os.path.exists(eeg_file):
                print(f"  ✅ Subject {i:02d}: EEG file found")
            else:
                print(f"  ⚠️ Subject {i:02d}: EEG file missing")
    
    print(f"📊 Found {len(subjects_found)} subjects: {subjects_found}")
    return len(subjects_found) > 0

def load_things_eeg_subject(subject_id, data_path="datasets/things_eeg"):
    """Load THINGS-EEG data for a specific subject"""
    print(f"📂 Loading THINGS-EEG subject {subject_id:02d}...")
    
    try:
        eeg_file = os.path.join(data_path, f"sub-{subject_id:02d}", "eeg", 
                               f"sub-{subject_id:02d}_task-rsvp_eeg.set")
        
        if not os.path.exists(eeg_file):
            print(f"❌ EEG file not found: {eeg_file}")
            return None, None
        
        # Load EEG data using MNE
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
        
        # Get data and events
        data = raw.get_data()  # Shape: [channels, timepoints]
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        print(f"  ✅ Loaded EEG data: {data.shape}")
        print(f"  📊 Found {len(events)} events")
        print(f"  📊 Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  📊 Channels: {len(raw.ch_names)}")
        
        return data, events
        
    except Exception as e:
        print(f"❌ Error loading THINGS-EEG data: {str(e)}")
        return None, None

def main():
    """Main validation function"""
    print("🔍 Real EEG Dataset Validation")
    print("=" * 40)
    
    # Validate THINGS-EEG
    if validate_things_eeg():
        print("✅ THINGS-EEG dataset validation passed")
        
        # Try to load first subject
        data, events = load_things_eeg_subject(1)
        if data is not None:
            print("✅ Successfully loaded sample data")
        else:
            print("⚠️ Could not load sample data")
    else:
        print("❌ THINGS-EEG dataset not found")
        print("💡 Please download from: https://osf.io/crn2h/")

if __name__ == "__main__":
    main()
'''
    
    with open('validate_real_datasets.py', 'w') as f:
        f.write(validation_script)
    
    print("  ✅ Created 'validate_real_datasets.py'")

def create_cross_subject_real_script():
    """Create cross-subject validation script for real datasets"""
    print("\n🔧 Creating cross-subject validation script for real data...")
    
    cross_validation_script = '''#!/usr/bin/env python3
# cross_subject_real_validation.py - Cross-subject validation with real datasets

import numpy as np
import os
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_trained_models():
    """Load our pre-trained models"""
    try:
        traditional_models = joblib.load('traditional_models.pkl')
        meta_model = joblib.load('meta_model.pkl')
        return traditional_models, meta_model
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        return None, None

def extract_simple_features(eeg_data):
    """Extract simple features from EEG data"""
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
                np.max(channel_data) - np.min(channel_data),
                np.percentile(channel_data, 75) - np.percentile(channel_data, 25)
            ])
        
        # Cross-channel correlations
        if trial.shape[0] >= 2:
            corr_matrix = np.corrcoef(trial)
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            trial_features.extend(upper_triangle[:10])  # First 10 correlations
        
        features.append(trial_features)
    
    return np.array(features, dtype=np.float64)

def test_real_dataset_subject(traditional_models, meta_model, features, labels):
    """Test subject with real dataset"""
    try:
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get predictions from available models
        predictions_list = []
        
        if 'svm' in traditional_models and traditional_models['svm'] is not None:
            svm_proba = traditional_models['svm'].predict_proba(features_scaled)
            predictions_list.append(svm_proba)
        
        if 'lr' in traditional_models and traditional_models['lr'] is not None:
            lr_proba = traditional_models['lr'].predict_proba(features_scaled)
            predictions_list.append(lr_proba)
        
        if len(predictions_list) == 0:
            return None
        
        # Combine for meta-model
        meta_features = np.hstack(predictions_list) if len(predictions_list) > 1 else predictions_list[0]
        
        # Final predictions
        final_predictions = meta_model.predict(meta_features)
        accuracy = accuracy_score(labels, final_predictions)
        
        return accuracy, final_predictions
        
    except Exception as e:
        print(f"❌ Error in testing: {str(e)}")
        return None, None

def main():
    """Main cross-subject validation with real data"""
    print("🚀 Cross-Subject Validation with Real EEG Data")
    print("=" * 50)
    
    # Load models
    traditional_models, meta_model = load_trained_models()
    if traditional_models is None:
        print("❌ Cannot proceed without trained models")
        return
    
    print("✅ Models loaded successfully")
    print("📝 Ready for real dataset validation")
    print()
    print("Next steps:")
    print("1. Download THINGS-EEG dataset from https://osf.io/crn2h/")
    print("2. Extract to 'datasets/things_eeg/'")
    print("3. Run validate_real_datasets.py")
    print("4. Implement specific loader for your chosen dataset")

if __name__ == "__main__":
    main()
'''
    
    with open('cross_subject_real_validation.py', 'w') as f:
        f.write(cross_validation_script)
    
    print("  ✅ Created 'cross_subject_real_validation.py'")

def main():
    """Main function"""
    print("🔗 Real EEG Visual Perception Datasets Guide")
    print("=" * 60)
    
    # Show information about real datasets
    download_things_eeg_info()
    download_eeg_imagenet_info()
    download_physionet_visual_info()
    
    # Create validation scripts
    create_dataset_validation_script()
    create_cross_subject_real_script()
    
    print("\n" + "=" * 60)
    print("📝 SUMMARY - Recommended Action Plan:")
    print("=" * 60)
    print()
    print("🎯 BEST OPTION: THINGS-EEG Dataset")
    print("  1. Go to: https://osf.io/crn2h/")
    print("  2. Download 2-3 subjects (instead of all 10) to start")
    print("  3. Each subject ~5GB, total ~15GB")
    print("  4. Extract to 'datasets/things_eeg/'")
    print("  5. Run: python validate_real_datasets.py")
    print("  6. Implement cross-subject validation")
    print()
    print("🔄 ALTERNATIVE: Use Your Current Dataset")
    print("  1. Keep using EP1.01.txt as baseline")
    print("  2. Test generalization with different preprocessing")
    print("  3. Focus on improving model architecture")
    print("  4. Document limitations in your research")
    print()
    print("💡 PRACTICAL APPROACH:")
    print("  1. Start with your current dataset (EP1.01.txt)")
    print("  2. Download 1-2 subjects from THINGS-EEG for validation")
    print("  3. Compare performance across datasets")
    print("  4. Document domain transfer capabilities")
    print()
    print("✅ Scripts created:")
    print("  - validate_real_datasets.py")
    print("  - cross_subject_real_validation.py")

if __name__ == "__main__":
    main()

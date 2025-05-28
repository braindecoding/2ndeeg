#!/usr/bin/env python3
# download_things_eeg.py - Helper script to download THINGS-EEG dataset

import os
import urllib.request
import zipfile
from pathlib import Path

def create_dataset_directory():
    """Create dataset directory structure"""
    print("📁 Creating dataset directory...")
    
    dataset_dir = Path("datasets/things_eeg")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Created directory: {dataset_dir}")
    return dataset_dir

def show_download_instructions():
    """Show detailed download instructions for THINGS-EEG"""
    print("📥 THINGS-EEG Dataset Download Instructions")
    print("=" * 60)
    print()
    print("🔗 Official Download Link:")
    print("   https://osf.io/crn2h/")
    print()
    print("📋 Dataset Information:")
    print("   - 10 subjects viewing real images")
    print("   - 22,248 images from THINGS database")
    print("   - 63 EEG channels, 1000 Hz sampling rate")
    print("   - Total size: ~50GB (full dataset)")
    print("   - Per subject: ~5GB")
    print()
    print("💡 Recommended Download Strategy:")
    print("   1. Start with 2-3 subjects for testing")
    print("   2. Download subjects individually")
    print("   3. Each subject is ~5GB")
    print()
    print("📁 Expected Structure After Download:")
    print("   datasets/things_eeg/")
    print("   ├── sub-01/")
    print("   │   ├── eeg/")
    print("   │   │   └── sub-01_task-rsvp_eeg.set")
    print("   │   └── beh/")
    print("   │       └── sub-01_task-rsvp_beh.tsv")
    print("   ├── sub-02/")
    print("   └── ...")
    print()
    print("🚀 Quick Start Steps:")
    print("   1. Go to: https://osf.io/crn2h/")
    print("   2. Click on individual subject folders (sub-01, sub-02, etc.)")
    print("   3. Download 'Download as zip' for each subject")
    print("   4. Extract each zip to datasets/things_eeg/")
    print("   5. Run validation script")

def download_sample_subject():
    """Attempt to download a sample subject (if available)"""
    print("\n🔄 Attempting to download sample data...")
    
    # Note: This is a placeholder - actual download would require
    # proper authentication and direct links from OSF
    print("⚠️ Automatic download not available")
    print("💡 Manual download required from OSF")
    
    return False

def validate_downloaded_data():
    """Validate if THINGS-EEG data has been downloaded"""
    print("\n🔍 Validating downloaded THINGS-EEG data...")
    
    dataset_dir = Path("datasets/things_eeg")
    
    if not dataset_dir.exists():
        print("❌ Dataset directory not found")
        return False, []
    
    subjects_found = []
    
    for i in range(1, 11):  # Check subjects 01-10
        subject_dir = dataset_dir / f"sub-{i:02d}"
        
        if subject_dir.exists():
            eeg_file = subject_dir / "eeg" / f"sub-{i:02d}_task-rsvp_eeg.set"
            beh_file = subject_dir / "beh" / f"sub-{i:02d}_task-rsvp_beh.tsv"
            
            if eeg_file.exists():
                subjects_found.append(i)
                print(f"  ✅ Subject {i:02d}: EEG file found")
                
                if beh_file.exists():
                    print(f"    ✅ Behavioral file found")
                else:
                    print(f"    ⚠️ Behavioral file missing")
            else:
                print(f"  ❌ Subject {i:02d}: EEG file missing")
    
    if len(subjects_found) >= 2:
        print(f"\n✅ Validation passed: {len(subjects_found)} subjects found")
        print(f"📊 Available subjects: {subjects_found}")
        return True, subjects_found
    else:
        print(f"\n❌ Validation failed: Only {len(subjects_found)} subjects found")
        print("💡 Need at least 2 subjects for cross-subject validation")
        return False, subjects_found

def create_test_with_current_data():
    """Create a test using current EP1.01.txt data"""
    print("\n🔄 Alternative: Test with current data...")
    
    current_data = Path("Data/EP1.01.txt")
    
    if current_data.exists():
        print("✅ Current dataset (EP1.01.txt) found")
        print("💡 You can proceed with cross-subject validation using:")
        print("   1. Your current dataset as baseline")
        print("   2. Synthetic alternative subjects for testing")
        print("   3. Download THINGS-EEG later for real validation")
        
        return True
    else:
        print("❌ Current dataset not found")
        return False

def main():
    """Main function"""
    print("🚀 THINGS-EEG Dataset Download Helper")
    print("=" * 50)
    
    # Create dataset directory
    dataset_dir = create_dataset_directory()
    
    # Show download instructions
    show_download_instructions()
    
    # Validate if data already exists
    is_valid, subjects = validate_downloaded_data()
    
    if is_valid:
        print("\n🎉 THINGS-EEG dataset is ready!")
        print("✅ You can now run: python3 things_eeg_cross_validation.py")
    else:
        print("\n📋 Next Steps:")
        print("=" * 30)
        print("1. Manual Download (Recommended):")
        print("   - Go to: https://osf.io/crn2h/")
        print("   - Download 2-3 subjects (~15GB)")
        print("   - Extract to datasets/things_eeg/")
        print("   - Run this script again to validate")
        print()
        print("2. Alternative - Use Current Data:")
        print("   - Continue with EP1.01.txt dataset")
        print("   - Use synthetic subjects for testing")
        print("   - Run: python3 test_cross_subject_synthetic.py")
        print()
        print("3. Check Current Data:")
        
        if create_test_with_current_data():
            print("   ✅ Current data available for testing")
        else:
            print("   ❌ No data available")
    
    print("\n📝 Summary:")
    print(f"   Dataset directory: {dataset_dir}")
    print(f"   THINGS-EEG subjects found: {len(subjects)}")
    print(f"   Ready for validation: {'Yes' if is_valid else 'No'}")

if __name__ == "__main__":
    main()

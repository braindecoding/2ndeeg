#!/usr/bin/env python3
# validate_things_eeg.py - Validate downloaded THINGS-EEG data

import os
from pathlib import Path

def validate_things_eeg():
    """Validate THINGS-EEG dataset structure"""
    print("🔍 Validating THINGS-EEG dataset...")
    
    dataset_dir = Path("datasets/things_eeg")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("💡 Please create directory and download data")
        return False, []
    
    subjects_found = []
    
    for i in range(1, 11):  # Check subjects 01-10
        subject_dir = dataset_dir / f"sub-{i:02d}"
        
        if subject_dir.exists():
            print(f"📁 Found subject directory: sub-{i:02d}")
            
            # Check for EEG file
            eeg_file = subject_dir / "eeg" / f"sub-{i:02d}_task-rsvp_eeg.set"
            if eeg_file.exists():
                file_size = eeg_file.stat().st_size / (1024**3)  # GB
                subjects_found.append(i)
                print(f"  ✅ EEG file found ({file_size:.1f} GB)")
            else:
                print(f"  ❌ EEG file missing: {eeg_file}")
            
            # Check for behavioral file
            beh_file = subject_dir / "beh" / f"sub-{i:02d}_task-rsvp_beh.tsv"
            if beh_file.exists():
                print(f"  ✅ Behavioral file found")
            else:
                print(f"  ⚠️ Behavioral file missing: {beh_file}")
    
    print(f"\n📊 Summary:")
    print(f"  Subjects found: {len(subjects_found)}")
    print(f"  Subject IDs: {subjects_found}")
    
    if len(subjects_found) >= 2:
        print(f"  ✅ Sufficient for cross-subject validation")
        return True, subjects_found
    else:
        print(f"  ❌ Need at least 2 subjects for validation")
        return False, subjects_found

def main():
    """Main validation function"""
    print("🔍 THINGS-EEG Dataset Validation")
    print("=" * 40)
    
    is_valid, subjects = validate_things_eeg()
    
    if is_valid:
        print(f"\n🎉 Dataset validation successful!")
        print(f"✅ Ready to run cross-subject validation")
        print(f"🚀 Next step: python3 things_eeg_cross_validation.py")
    else:
        print(f"\n❌ Dataset validation failed")
        print(f"💡 Please download THINGS-EEG data first")
        print(f"🔗 Go to: https://osf.io/crn2h/")

if __name__ == "__main__":
    main()

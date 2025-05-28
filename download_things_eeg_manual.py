#!/usr/bin/env python3
# download_things_eeg_manual.py - Manual download guide for THINGS-EEG dataset

import os
import webbrowser
from pathlib import Path

def open_download_page():
    """Open THINGS-EEG download page in browser"""
    print("🌐 Opening THINGS-EEG download page...")
    
    url = "https://osf.io/crn2h/"
    
    try:
        webbrowser.open(url)
        print(f"✅ Opened: {url}")
        return True
    except Exception as e:
        print(f"❌ Could not open browser: {str(e)}")
        print(f"🔗 Please manually go to: {url}")
        return False

def show_detailed_instructions():
    """Show detailed download and setup instructions"""
    print("\n📥 THINGS-EEG Dataset Download Instructions")
    print("=" * 60)
    print()
    print("🎯 GOAL: Download 2-3 subjects for cross-subject validation")
    print("📊 Each subject: ~5GB")
    print("⏱️ Total download time: ~30-60 minutes (depending on internet)")
    print()
    print("🔗 Download URL: https://osf.io/crn2h/")
    print()
    print("📋 Step-by-Step Instructions:")
    print("=" * 40)
    print()
    print("STEP 1: Navigate to OSF")
    print("  1. Go to: https://osf.io/crn2h/")
    print("  2. You'll see the THINGS-EEG project page")
    print("  3. Look for 'Files' section")
    print()
    print("STEP 2: Choose Subjects to Download")
    print("  🎯 RECOMMENDED: Start with sub-01, sub-02, sub-03")
    print("  1. Click on 'sub-01' folder")
    print("  2. Click 'Download as zip' button")
    print("  3. Repeat for sub-02 and sub-03")
    print()
    print("STEP 3: Extract Files")
    print("  1. Extract each zip file")
    print("  2. Move extracted folders to: datasets/things_eeg/")
    print("  3. Final structure should be:")
    print("     datasets/things_eeg/")
    print("     ├── sub-01/")
    print("     │   ├── eeg/")
    print("     │   │   └── sub-01_task-rsvp_eeg.set")
    print("     │   └── beh/")
    print("     │       └── sub-01_task-rsvp_beh.tsv")
    print("     ├── sub-02/")
    print("     └── sub-03/")
    print()
    print("STEP 4: Verify Download")
    print("  1. Run: python3 validate_things_eeg.py")
    print("  2. Should show 3 subjects found")
    print()
    print("⚠️ IMPORTANT NOTES:")
    print("  - Each subject zip is ~5GB")
    print("  - Make sure you have ~20GB free space")
    print("  - Download may take 30-60 minutes")
    print("  - You can download more subjects later")

def create_validation_script():
    """Create validation script for downloaded data"""
    print("\n🔧 Creating validation script...")
    
    validation_code = '''#!/usr/bin/env python3
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
    
    print(f"\\n📊 Summary:")
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
        print(f"\\n🎉 Dataset validation successful!")
        print(f"✅ Ready to run cross-subject validation")
        print(f"🚀 Next step: python3 things_eeg_cross_validation.py")
    else:
        print(f"\\n❌ Dataset validation failed")
        print(f"💡 Please download THINGS-EEG data first")
        print(f"🔗 Go to: https://osf.io/crn2h/")

if __name__ == "__main__":
    main()
'''
    
    with open('validate_things_eeg.py', 'w') as f:
        f.write(validation_code)
    
    print("✅ Created: validate_things_eeg.py")

def create_download_checklist():
    """Create a downloadable checklist"""
    print("\n📋 Creating download checklist...")
    
    checklist = """
THINGS-EEG Dataset Download Checklist
====================================

□ Step 1: Prepare
  □ Check available disk space (need ~20GB)
  □ Ensure stable internet connection
  □ Create datasets/things_eeg/ directory

□ Step 2: Download from OSF
  □ Go to: https://osf.io/crn2h/
  □ Download sub-01.zip (~5GB)
  □ Download sub-02.zip (~5GB) 
  □ Download sub-03.zip (~5GB)
  □ Optional: Download more subjects

□ Step 3: Extract Files
  □ Extract sub-01.zip to datasets/things_eeg/sub-01/
  □ Extract sub-02.zip to datasets/things_eeg/sub-02/
  □ Extract sub-03.zip to datasets/things_eeg/sub-03/
  □ Verify folder structure

□ Step 4: Validate
  □ Run: python3 validate_things_eeg.py
  □ Should show 3 subjects found
  □ All EEG files present

□ Step 5: Run Analysis
  □ Run: python3 things_eeg_cross_validation.py
  □ Check results and visualizations

Expected File Structure:
datasets/things_eeg/
├── sub-01/
│   ├── eeg/
│   │   └── sub-01_task-rsvp_eeg.set
│   └── beh/
│       └── sub-01_task-rsvp_beh.tsv
├── sub-02/
└── sub-03/

Troubleshooting:
- If download is slow: Try downloading one subject at a time
- If extraction fails: Check available disk space
- If validation fails: Check file paths and names
- If analysis fails: Ensure all dependencies installed
"""
    
    with open('things_eeg_download_checklist.txt', 'w') as f:
        f.write(checklist)
    
    print("✅ Created: things_eeg_download_checklist.txt")

def check_current_status():
    """Check current download status"""
    print("\n🔍 Checking current download status...")
    
    dataset_dir = Path("datasets/things_eeg")
    
    if not dataset_dir.exists():
        print("❌ Dataset directory not found")
        print("💡 Directory will be created")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dataset_dir}")
    else:
        print(f"✅ Dataset directory exists: {dataset_dir}")
    
    # Check for any existing subjects
    subjects_found = []
    for i in range(1, 11):
        subject_dir = dataset_dir / f"sub-{i:02d}"
        if subject_dir.exists():
            subjects_found.append(i)
    
    if len(subjects_found) > 0:
        print(f"📊 Existing subjects found: {subjects_found}")
    else:
        print("📊 No subjects found yet")
    
    return len(subjects_found)

def main():
    """Main function"""
    print("🚀 THINGS-EEG Dataset Manual Download Helper")
    print("=" * 60)
    
    # Check current status
    existing_subjects = check_current_status()
    
    if existing_subjects >= 2:
        print(f"\n🎉 You already have {existing_subjects} subjects!")
        print("✅ You can proceed with validation")
        print("🚀 Run: python3 validate_things_eeg.py")
    else:
        # Show instructions
        show_detailed_instructions()
        
        # Create helper scripts
        create_validation_script()
        create_download_checklist()
        
        # Open download page
        print("\n🌐 Opening download page...")
        open_download_page()
        
        print("\n📋 SUMMARY - What to do next:")
        print("=" * 40)
        print("1. ✅ Dataset directory created")
        print("2. ✅ Validation script created")
        print("3. ✅ Download checklist created")
        print("4. 🌐 Download page opened in browser")
        print()
        print("🎯 YOUR NEXT STEPS:")
        print("1. Download 3 subjects from: https://osf.io/crn2h/")
        print("2. Extract to datasets/things_eeg/")
        print("3. Run: python3 validate_things_eeg.py")
        print("4. Run: python3 things_eeg_cross_validation.py")
        print()
        print("📄 Files created:")
        print("  - validate_things_eeg.py")
        print("  - things_eeg_download_checklist.txt")

if __name__ == "__main__":
    main()

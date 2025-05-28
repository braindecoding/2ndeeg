#!/usr/bin/env python3
# check_session_files.py - Check both session files

import os
import mne
from pathlib import Path

def check_fif_file(filepath):
    """Check a single .fif file"""
    print(f"🔍 Checking {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        return False
    
    # Check file size
    file_size = os.path.getsize(filepath) / (1024**2)  # MB
    print(f"  📊 File size: {file_size:.1f} MB")
    
    try:
        # Try to load the file
        raw = mne.io.read_raw_fif(str(filepath), preload=False, verbose=False)
        
        print(f"  ✅ File loaded successfully")
        print(f"  📊 Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  📊 Channels: {len(raw.ch_names)}")
        print(f"  📊 Duration: {raw.times[-1]:.1f} seconds")
        print(f"  📊 Channel types: {set(raw.get_channel_types())}")
        print(f"  📊 First 10 channels: {raw.ch_names[:10]}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading file: {str(e)}")
        return False

def main():
    """Check both session files"""
    print("🔍 Checking Visual EEG Session Files")
    print("=" * 50)
    
    # Check all possible session files
    session_files = [
        "datasets/subj01_session1_eeg.fif",
        "datasets/subj01_session1_eeg.fif copy",
        "datasets/subj01_session2_eeg.fif"
    ]
    
    valid_files = []
    
    for filepath in session_files:
        if check_fif_file(filepath):
            valid_files.append(filepath)
        print()
    
    print(f"📊 Summary:")
    print(f"  Total files checked: {len(session_files)}")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Valid file paths: {valid_files}")
    
    if len(valid_files) >= 2:
        print("✅ Sufficient files for cross-session analysis")
    elif len(valid_files) == 1:
        print("⚠️ Only one valid file - can still proceed with single session")
    else:
        print("❌ No valid files found")

if __name__ == "__main__":
    main()

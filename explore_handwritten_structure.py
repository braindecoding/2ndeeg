#!/usr/bin/env python3
# explore_handwritten_structure.py - Deep exploration of handwritten dataset structure

import numpy as np
from scipy.io import loadmat
from pathlib import Path

def explore_nested_structure(data, name="", level=0):
    """Recursively explore nested data structure"""
    indent = "  " * level
    
    if isinstance(data, np.ndarray):
        print(f"{indent}{name}: numpy.ndarray")
        print(f"{indent}  Shape: {data.shape}")
        print(f"{indent}  Dtype: {data.dtype}")
        
        if data.dtype.names:  # Structured array
            print(f"{indent}  Fields: {data.dtype.names}")
            if data.size > 0:
                for field in data.dtype.names:
                    field_data = data[field][0] if data.shape == (1, 1) else data[field]
                    explore_nested_structure(field_data, f"{name}.{field}", level + 1)
        else:
            if data.size > 0 and data.size < 20:
                print(f"{indent}  Values: {data.flatten()}")
            elif data.size > 0:
                print(f"{indent}  Sample values: {data.flatten()[:10]}")
                print(f"{indent}  Min: {np.min(data)}, Max: {np.max(data)}")
    else:
        print(f"{indent}{name}: {type(data)} = {data}")

def deep_explore_handwritten():
    """Deep exploration of handwritten dataset"""
    print("🔍 Deep Exploration of Handwritten Character Dataset")
    print("=" * 60)
    
    mat_file = "datasets/Handwritten character dataset/S01.mat"
    
    if not Path(mat_file).exists():
        print(f"❌ File not found: {mat_file}")
        return
    
    # Load data
    data = loadmat(mat_file)
    
    # Remove metadata
    data_keys = [k for k in data.keys() if not k.startswith('__')]
    
    for key in data_keys:
        print(f"\n{'='*50}")
        print(f"🔍 EXPLORING KEY: {key}")
        print(f"{'='*50}")
        
        explore_nested_structure(data[key], key)
        
        # Special handling for nested structures
        if hasattr(data[key], 'shape') and data[key].shape == (1, 1):
            nested = data[key][0, 0]
            
            if hasattr(nested, 'dtype') and nested.dtype.names:
                print(f"\n📊 Detailed analysis of {key}:")
                
                for field in nested.dtype.names:
                    field_data = nested[field][0]
                    print(f"\n  🔍 Field: {field}")
                    print(f"    Type: {type(field_data)}")
                    print(f"    Shape: {field_data.shape if hasattr(field_data, 'shape') else 'N/A'}")
                    
                    if hasattr(field_data, 'shape'):
                        if field_data.ndim == 2:
                            print(f"    2D Array: {field_data.shape[0]} x {field_data.shape[1]}")
                            if field_data.shape[0] > 1000 and field_data.shape[1] > 10:
                                print(f"    ✅ Potential EEG data: {field_data.shape}")
                                print(f"    Sample values: {field_data[:3, :5]}")
                        elif field_data.ndim == 1:
                            print(f"    1D Array: {field_data.shape[0]} elements")
                            if field_data.shape[0] < 1000:
                                unique_vals = np.unique(field_data)
                                print(f"    Unique values: {unique_vals[:20]}")
                                if len(unique_vals) < 50:
                                    print(f"    ✅ Potential labels/markers")

def extract_actual_eeg_data():
    """Try to extract actual EEG data"""
    print(f"\n🔧 Attempting to extract actual EEG data...")
    
    mat_file = "datasets/Handwritten character dataset/S01.mat"
    data = loadmat(mat_file)
    
    data_keys = [k for k in data.keys() if not k.startswith('__')]
    
    all_eeg_data = []
    all_labels = []
    
    for key in data_keys:
        print(f"\n📂 Processing {key}...")
        
        if hasattr(data[key], 'shape') and data[key].shape == (1, 1):
            nested = data[key][0, 0]
            
            if hasattr(nested, 'dtype') and nested.dtype.names:
                # Look for BrainVisionRDA_data
                if 'BrainVisionRDA_data' in nested.dtype.names:
                    eeg_field = nested['BrainVisionRDA_data'][0]
                    print(f"  EEG field shape: {eeg_field.shape}")
                    print(f"  EEG field type: {type(eeg_field)}")
                    
                    # Check if it's a reference to another array
                    if hasattr(eeg_field, 'shape') and len(eeg_field.shape) == 2:
                        if eeg_field.shape[0] > 1000:  # Likely timepoints x channels
                            print(f"  ✅ Found EEG data: {eeg_field.shape}")
                            all_eeg_data.append(eeg_field)
                            
                            # Create simple labels based on key
                            if 'round01' in key:
                                labels = np.zeros(eeg_field.shape[0])
                            else:
                                labels = np.ones(eeg_field.shape[0])
                            
                            all_labels.append(labels)
                            
                            print(f"  📊 Sample EEG values: {eeg_field[:3, :3]}")
                
                # Look for marker data
                for marker_field in ['EyeblockMarker_data', 'ParadigmMarker_data']:
                    if marker_field in nested.dtype.names:
                        marker_data = nested[marker_field][0]
                        print(f"  {marker_field} shape: {marker_data.shape}")
                        if hasattr(marker_data, 'flatten'):
                            unique_markers = np.unique(marker_data.flatten())
                            print(f"  Unique markers: {unique_markers}")
    
    if len(all_eeg_data) > 0:
        print(f"\n✅ Successfully extracted EEG data from {len(all_eeg_data)} sessions")
        for i, eeg in enumerate(all_eeg_data):
            print(f"  Session {i+1}: {eeg.shape}")
        
        # Combine all data
        combined_eeg = np.vstack(all_eeg_data)
        combined_labels = np.concatenate(all_labels)
        
        print(f"\n📊 Combined dataset:")
        print(f"  EEG shape: {combined_eeg.shape}")
        print(f"  Labels shape: {combined_labels.shape}")
        print(f"  Label distribution: {np.bincount(combined_labels.astype(int))}")
        
        # Save for later use
        np.save('handwritten_eeg_data.npy', combined_eeg)
        np.save('handwritten_labels.npy', combined_labels)
        print(f"\n💾 Saved data as 'handwritten_eeg_data.npy' and 'handwritten_labels.npy'")
        
        return combined_eeg, combined_labels
    else:
        print(f"\n❌ No EEG data found")
        return None, None

def analyze_handwritten_characteristics():
    """Analyze characteristics of the handwritten dataset"""
    print(f"\n📊 Analyzing Handwritten Dataset Characteristics")
    print("=" * 50)
    
    try:
        eeg_data = np.load('handwritten_eeg_data.npy')
        labels = np.load('handwritten_labels.npy')
        
        print(f"📊 Dataset Overview:")
        print(f"  EEG Data Shape: {eeg_data.shape}")
        print(f"  Labels Shape: {labels.shape}")
        print(f"  Data Type: {eeg_data.dtype}")
        print(f"  Memory Usage: {eeg_data.nbytes / (1024**2):.1f} MB")
        
        print(f"\n📊 Signal Characteristics:")
        print(f"  Min Value: {np.min(eeg_data):.2f}")
        print(f"  Max Value: {np.max(eeg_data):.2f}")
        print(f"  Mean Value: {np.mean(eeg_data):.2f}")
        print(f"  Std Value: {np.std(eeg_data):.2f}")
        
        print(f"\n📊 Label Analysis:")
        unique_labels = np.unique(labels)
        print(f"  Unique Labels: {unique_labels}")
        print(f"  Label Counts: {np.bincount(labels.astype(int))}")
        
        # Sampling rate estimation
        if eeg_data.shape[0] > 1000:
            print(f"\n📊 Estimated Properties:")
            print(f"  Timepoints: {eeg_data.shape[0]}")
            print(f"  Channels: {eeg_data.shape[1]}")
            print(f"  Estimated Duration: ~{eeg_data.shape[0]/1000:.1f} seconds (assuming 1000 Hz)")
        
        return True
        
    except FileNotFoundError:
        print("❌ No saved data found. Run extraction first.")
        return False

def main():
    """Main exploration function"""
    print("🔍 Handwritten Character Dataset Deep Exploration")
    print("=" * 60)
    
    # Step 1: Deep structure exploration
    deep_explore_handwritten()
    
    # Step 2: Extract actual EEG data
    eeg_data, labels = extract_actual_eeg_data()
    
    # Step 3: Analyze characteristics
    if eeg_data is not None:
        analyze_handwritten_characteristics()
        
        print(f"\n✅ Exploration completed successfully!")
        print(f"📊 Found EEG data: {eeg_data.shape}")
        print(f"📊 Found labels: {labels.shape}")
        
        print(f"\n🚀 Next steps:")
        print("1. Use extracted data for cross-subject validation")
        print("2. Compare with existing digit classification models")
        print("3. Analyze handwriting-specific EEG patterns")
    else:
        print(f"\n❌ Could not extract EEG data")

if __name__ == "__main__":
    main()

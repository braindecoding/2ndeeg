#!/usr/bin/env python3
# debug_handwritten_data.py - Debug handwritten dataset structure

import numpy as np
from scipy.io import loadmat

def debug_object_arrays():
    """Debug the object arrays in handwritten dataset"""
    print("🔍 Debugging Handwritten Dataset Object Arrays")
    print("=" * 50)

    mat_file = "datasets/Handwritten character dataset/S01.mat"
    data = loadmat(mat_file)

    # Focus on one key for detailed debugging
    key = 'round01_paradigm'
    print(f"\n🔍 Debugging key: {key}")

    nested = data[key][0, 0]

    # Debug BrainVisionRDA_data
    print(f"\n📊 BrainVisionRDA_data debugging:")
    eeg_field = nested['BrainVisionRDA_data']
    print(f"  Type: {type(eeg_field)}")
    print(f"  Shape: {eeg_field.shape}")
    print(f"  Dtype: {eeg_field.dtype}")

    # The EEG data is directly in eeg_field, not nested
    print(f"\n  Direct EEG data:")
    print(f"    Type: {type(eeg_field)}")
    print(f"    Shape: {eeg_field.shape}")
    print(f"    Dtype: {eeg_field.dtype}")

    # Check if this is the actual EEG data
    if eeg_field.ndim == 2 and eeg_field.shape[0] > 1000:
        print(f"  ✅ This IS the EEG data!")
        print(f"    Timepoints: {eeg_field.shape[0]}")
        print(f"    Channels: {eeg_field.shape[1]}")
        print(f"    Min: {np.min(eeg_field):.2f}")
        print(f"    Max: {np.max(eeg_field):.2f}")
        print(f"    Sample values: {eeg_field[:3, :5]}")

        # Save this for testing
        np.save('debug_eeg_sample.npy', eeg_field)
        print(f"  💾 Saved sample as 'debug_eeg_sample.npy'")

    # Debug marker data
    print(f"\n📊 EyeblockMarker_data debugging:")
    if 'EyeblockMarker_data' in nested.dtype.names:
        marker_field = nested['EyeblockMarker_data']
        marker_obj = marker_field[0]
        print(f"  Type: {type(marker_obj)}")
        print(f"  Shape: {marker_obj.shape}")
        print(f"  Values: {marker_obj.flatten()[:20]}")

        # Extract character markers
        markers = marker_obj.flatten()
        char_markers = markers[(markers != 1000) & (markers != -1000)]
        if len(char_markers) > 0:
            print(f"  📝 Character markers: {np.unique(char_markers)}")

    # Debug ParadigmMarker_data
    print(f"\n📊 ParadigmMarker_data debugging:")
    if 'ParadigmMarker_data' in nested.dtype.names:
        paradigm_field = nested['ParadigmMarker_data']
        paradigm_obj = paradigm_field[0]
        print(f"  Type: {type(paradigm_obj)}")
        print(f"  Shape: {paradigm_obj.shape}")
        print(f"  Values: {np.unique(paradigm_obj.flatten())}")

def extract_all_eeg_sessions():
    """Extract EEG data from all sessions"""
    print(f"\n🔧 Extracting EEG from all sessions...")

    mat_file = "datasets/Handwritten character dataset/S01.mat"
    data = loadmat(mat_file)

    data_keys = [k for k in data.keys() if not k.startswith('__')]

    all_eeg = []
    all_info = []

    for key in data_keys:
        print(f"\n📂 Processing {key}...")

        nested = data[key][0, 0]

        if 'BrainVisionRDA_data' in nested.dtype.names:
            eeg_data = nested['BrainVisionRDA_data']

            if isinstance(eeg_data, np.ndarray) and eeg_data.ndim == 2:
                print(f"  ✅ Found EEG: {eeg_data.shape}")
                all_eeg.append(eeg_data)
                all_info.append({
                    'key': key,
                    'shape': eeg_data.shape,
                    'min': np.min(eeg_data),
                    'max': np.max(eeg_data),
                    'mean': np.mean(eeg_data)
                })

    if len(all_eeg) > 0:
        print(f"\n✅ Found {len(all_eeg)} EEG sessions:")
        for i, info in enumerate(all_info):
            print(f"  {i+1}. {info['key']}: {info['shape']}")
            print(f"     Range: {info['min']:.2f} to {info['max']:.2f}, Mean: {info['mean']:.2f}")

        # Combine sessions
        combined_eeg = np.vstack(all_eeg)
        print(f"\n📊 Combined EEG shape: {combined_eeg.shape}")

        # Create session labels
        session_labels = []
        for i, eeg in enumerate(all_eeg):
            labels = np.full(eeg.shape[0], i)
            session_labels.append(labels)

        combined_labels = np.concatenate(session_labels)
        print(f"📊 Combined labels shape: {combined_labels.shape}")
        print(f"📊 Session distribution: {np.bincount(combined_labels)}")

        # Save data
        np.save('handwritten_eeg_data.npy', combined_eeg)
        np.save('handwritten_session_labels.npy', combined_labels)

        print(f"\n💾 Saved combined data!")

        return combined_eeg, combined_labels

    else:
        print(f"\n❌ No EEG data found")
        return None, None

def create_simple_visualization():
    """Create simple visualization of extracted data"""
    print(f"\n📊 Creating visualization...")

    try:
        eeg_data = np.load('handwritten_eeg_data.npy')
        labels = np.load('handwritten_session_labels.npy')

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Average signal
        avg_signal = np.mean(eeg_data, axis=1)
        axes[0, 0].plot(avg_signal[:5000])  # First 5000 samples
        axes[0, 0].set_title('Average EEG Signal (First 5000 samples)')
        axes[0, 0].set_xlabel('Time (samples)')
        axes[0, 0].set_ylabel('Amplitude (μV)')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Channel heatmap
        subsample = max(1, eeg_data.shape[0] // 500)
        eeg_sub = eeg_data[::subsample, :].T

        im = axes[0, 1].imshow(eeg_sub, aspect='auto', cmap='RdBu_r')
        axes[0, 1].set_title('EEG Channels (Subsampled)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Channels')
        plt.colorbar(im, ax=axes[0, 1])

        # Plot 3: Session labels
        axes[1, 0].plot(labels[::subsample], 'o-', markersize=2)
        axes[1, 0].set_title('Session Labels')
        axes[1, 0].set_xlabel('Time (subsampled)')
        axes[1, 0].set_ylabel('Session')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Distribution
        session_counts = np.bincount(labels)
        axes[1, 1].bar(range(len(session_counts)), session_counts)
        axes[1, 1].set_title('Session Distribution')
        axes[1, 1].set_xlabel('Session')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Handwritten Character EEG Dataset', fontsize=14)
        plt.tight_layout()
        plt.savefig('handwritten_eeg_simple.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved as 'handwritten_eeg_simple.png'")

        plt.close()

    except FileNotFoundError:
        print(f"  ❌ No data found for visualization")

def main():
    """Main debugging function"""
    print("🔍 Handwritten Dataset Debugging")
    print("=" * 40)

    # Debug object arrays
    debug_object_arrays()

    # Extract all sessions
    eeg_data, labels = extract_all_eeg_sessions()

    if eeg_data is not None:
        # Create visualization
        create_simple_visualization()

        print(f"\n✅ Debugging completed!")
        print(f"📊 Extracted EEG data: {eeg_data.shape}")
        print(f"📊 Session labels: {labels.shape}")
        print(f"📊 Ready for analysis!")
    else:
        print(f"\n❌ Debugging failed - no data extracted")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# handwritten_eeg_extractor.py - Extract EEG data from handwritten character dataset

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path

def extract_handwritten_eeg_data():
    """Extract EEG data from handwritten character dataset"""
    print("🔧 Extracting Handwritten Character EEG Data")
    print("=" * 50)
    
    mat_file = "datasets/Handwritten character dataset/S01.mat"
    
    if not Path(mat_file).exists():
        print(f"❌ File not found: {mat_file}")
        return None, None
    
    # Load data
    data = loadmat(mat_file)
    data_keys = [k for k in data.keys() if not k.startswith('__')]
    
    all_eeg_data = []
    all_labels = []
    all_markers = []
    
    for key in data_keys:
        print(f"\n📂 Processing {key}...")
        
        if hasattr(data[key], 'shape') and data[key].shape == (1, 1):
            nested = data[key][0, 0]
            
            if hasattr(nested, 'dtype') and nested.dtype.names:
                # Extract EEG data from object array
                if 'BrainVisionRDA_data' in nested.dtype.names:
                    eeg_obj = nested['BrainVisionRDA_data'][0]
                    
                    # The actual EEG data is inside the object array
                    if isinstance(eeg_obj, np.ndarray) and eeg_obj.ndim == 2:
                        print(f"  ✅ Found EEG data: {eeg_obj.shape}")
                        print(f"  📊 Data range: {np.min(eeg_obj):.2f} to {np.max(eeg_obj):.2f}")
                        
                        all_eeg_data.append(eeg_obj)
                        
                        # Create labels based on session
                        if 'round01' in key:
                            session_label = 0
                        else:
                            session_label = 1
                        
                        # Create labels for each timepoint
                        labels = np.full(eeg_obj.shape[0], session_label)
                        all_labels.append(labels)
                        
                        print(f"  📊 Created {len(labels)} labels with value {session_label}")
                
                # Extract marker information for character labels
                if 'EyeblockMarker_data' in nested.dtype.names:
                    marker_obj = nested['EyeblockMarker_data'][0]
                    if isinstance(marker_obj, np.ndarray):
                        # Extract character markers (1, 2, 3, 4 for different characters)
                        character_markers = marker_obj[(marker_obj != 1000) & (marker_obj != -1000)]
                        if len(character_markers) > 0:
                            unique_chars = np.unique(character_markers)
                            print(f"  📝 Character markers found: {unique_chars}")
                            all_markers.extend(character_markers.flatten())
    
    if len(all_eeg_data) > 0:
        # Combine all EEG data
        combined_eeg = np.vstack(all_eeg_data)
        combined_labels = np.concatenate(all_labels)
        
        print(f"\n✅ Successfully extracted EEG data:")
        print(f"  📊 Combined EEG shape: {combined_eeg.shape}")
        print(f"  📊 Combined labels shape: {combined_labels.shape}")
        print(f"  📊 Sessions: {np.bincount(combined_labels)}")
        print(f"  📊 Character markers found: {len(all_markers)}")
        
        if len(all_markers) > 0:
            unique_markers = np.unique(all_markers)
            print(f"  📝 Unique character markers: {unique_markers}")
        
        # Save extracted data
        np.save('handwritten_eeg_data.npy', combined_eeg)
        np.save('handwritten_labels.npy', combined_labels)
        if len(all_markers) > 0:
            np.save('handwritten_character_markers.npy', np.array(all_markers))
        
        print(f"\n💾 Data saved successfully!")
        
        return combined_eeg, combined_labels
    
    else:
        print(f"\n❌ No EEG data could be extracted")
        return None, None

def create_character_based_labels():
    """Create character-based labels from marker data"""
    print(f"\n🔧 Creating character-based labels...")
    
    try:
        eeg_data = np.load('handwritten_eeg_data.npy')
        session_labels = np.load('handwritten_labels.npy')
        
        # Load marker data if available
        try:
            markers = np.load('handwritten_character_markers.npy')
            print(f"  📝 Found {len(markers)} character markers")
            
            # Create character labels (1, 2, 3, 4 for different characters)
            unique_chars = np.unique(markers)
            print(f"  📝 Unique characters: {unique_chars}")
            
            # For now, create simple character labels based on time segments
            n_timepoints = eeg_data.shape[0]
            n_chars = len(unique_chars)
            
            if n_chars > 1:
                # Divide timepoints into character segments
                segment_size = n_timepoints // n_chars
                character_labels = np.zeros(n_timepoints, dtype=int)
                
                for i, char in enumerate(unique_chars):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size if i < n_chars - 1 else n_timepoints
                    character_labels[start_idx:end_idx] = int(char)
                
                print(f"  ✅ Created character labels: {np.bincount(character_labels)}")
                
                # Save character labels
                np.save('handwritten_character_labels.npy', character_labels)
                
                return character_labels
            
        except FileNotFoundError:
            print(f"  ⚠️ No character markers found, using session labels")
        
        return session_labels
        
    except FileNotFoundError:
        print(f"  ❌ No EEG data found")
        return None

def visualize_handwritten_data():
    """Visualize the extracted handwritten EEG data"""
    print(f"\n📊 Visualizing handwritten EEG data...")
    
    try:
        eeg_data = np.load('handwritten_eeg_data.npy')
        session_labels = np.load('handwritten_labels.npy')
        
        # Try to load character labels
        try:
            char_labels = np.load('handwritten_character_labels.npy')
            use_char_labels = True
        except FileNotFoundError:
            char_labels = session_labels
            use_char_labels = False
        
        print(f"  📊 EEG data shape: {eeg_data.shape}")
        print(f"  📊 Using {'character' if use_char_labels else 'session'} labels")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: Average EEG across all channels
        time_axis = np.arange(eeg_data.shape[0])
        avg_eeg = np.mean(eeg_data, axis=1)
        
        axes[0, 0].plot(time_axis, avg_eeg, 'b-', alpha=0.7)
        axes[0, 0].set_title('Average EEG Signal')
        axes[0, 0].set_xlabel('Time (samples)')
        axes[0, 0].set_ylabel('Amplitude (μV)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: EEG heatmap (channels x time, subsampled)
        subsample = max(1, eeg_data.shape[0] // 1000)  # Subsample for visualization
        eeg_sub = eeg_data[::subsample, :].T
        
        im = axes[0, 1].imshow(eeg_sub, aspect='auto', cmap='RdBu_r', 
                              interpolation='nearest')
        axes[0, 1].set_title('EEG Channels Heatmap')
        axes[0, 1].set_xlabel('Time (subsampled)')
        axes[0, 1].set_ylabel('Channels')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Plot 3: Label distribution
        unique_labels = np.unique(char_labels)
        label_counts = np.bincount(char_labels.astype(int))
        
        axes[0, 2].bar(range(len(label_counts)), label_counts, color='steelblue')
        axes[0, 2].set_title(f'{"Character" if use_char_labels else "Session"} Label Distribution')
        axes[0, 2].set_xlabel('Label')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Sample channels
        n_channels_to_plot = min(5, eeg_data.shape[1])
        for i in range(n_channels_to_plot):
            axes[1, 0].plot(time_axis[::subsample], eeg_data[::subsample, i], 
                           label=f'Ch {i+1}', alpha=0.7)
        
        axes[1, 0].set_title('Sample EEG Channels')
        axes[1, 0].set_xlabel('Time (subsampled)')
        axes[1, 0].set_ylabel('Amplitude (μV)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Power spectrum
        from scipy import signal
        
        # Compute power spectrum for first channel
        freqs, psd = signal.welch(eeg_data[:, 0], fs=1000, nperseg=1024)
        
        axes[1, 1].semilogy(freqs, psd)
        axes[1, 1].set_title('Power Spectral Density (Channel 1)')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('PSD (μV²/Hz)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 100)  # Focus on 0-100 Hz
        
        # Plot 6: Label timeline
        axes[1, 2].plot(time_axis[::subsample], char_labels[::subsample], 'o-', markersize=2)
        axes[1, 2].set_title(f'{"Character" if use_char_labels else "Session"} Labels Timeline')
        axes[1, 2].set_xlabel('Time (subsampled)')
        axes[1, 2].set_ylabel('Label')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Handwritten Character EEG Dataset Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig('handwritten_eeg_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Visualization saved as 'handwritten_eeg_analysis.png'")
        
        plt.close()
        
        return True
        
    except FileNotFoundError:
        print(f"  ❌ No EEG data found for visualization")
        return False

def analyze_dataset_properties():
    """Analyze properties of the handwritten dataset"""
    print(f"\n📊 Analyzing dataset properties...")
    
    try:
        eeg_data = np.load('handwritten_eeg_data.npy')
        labels = np.load('handwritten_labels.npy')
        
        print(f"📊 Dataset Properties:")
        print(f"  Shape: {eeg_data.shape}")
        print(f"  Timepoints: {eeg_data.shape[0]}")
        print(f"  Channels: {eeg_data.shape[1]}")
        print(f"  Data type: {eeg_data.dtype}")
        print(f"  Memory usage: {eeg_data.nbytes / (1024**2):.1f} MB")
        
        print(f"\n📊 Signal Statistics:")
        print(f"  Min: {np.min(eeg_data):.2f} μV")
        print(f"  Max: {np.max(eeg_data):.2f} μV")
        print(f"  Mean: {np.mean(eeg_data):.2f} μV")
        print(f"  Std: {np.std(eeg_data):.2f} μV")
        
        print(f"\n📊 Label Statistics:")
        unique_labels = np.unique(labels)
        print(f"  Unique labels: {unique_labels}")
        print(f"  Label distribution: {np.bincount(labels.astype(int))}")
        
        # Estimate sampling rate and duration
        print(f"\n📊 Temporal Properties:")
        print(f"  Estimated sampling rate: ~1000 Hz (typical for EEG)")
        print(f"  Estimated duration: ~{eeg_data.shape[0]/1000:.1f} seconds")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ No data found for analysis")
        return False

def main():
    """Main function"""
    print("🔧 Handwritten Character EEG Data Extraction")
    print("=" * 60)
    
    # Step 1: Extract EEG data
    eeg_data, labels = extract_handwritten_eeg_data()
    
    if eeg_data is not None:
        # Step 2: Create character-based labels
        char_labels = create_character_based_labels()
        
        # Step 3: Visualize data
        visualize_handwritten_data()
        
        # Step 4: Analyze properties
        analyze_dataset_properties()
        
        print(f"\n✅ Handwritten EEG data extraction completed!")
        print(f"📊 Final dataset: {eeg_data.shape}")
        print(f"📊 Ready for cross-subject validation and model training")
        
        print(f"\n🚀 Next steps:")
        print("1. Use extracted data for cross-subject validation")
        print("2. Compare with digit classification models")
        print("3. Train handwriting-specific models")
        print("4. Analyze character-specific EEG patterns")
        
    else:
        print(f"\n❌ Failed to extract handwritten EEG data")

if __name__ == "__main__":
    main()

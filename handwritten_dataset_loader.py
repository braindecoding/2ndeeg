#!/usr/bin/env python3
# handwritten_dataset_loader.py - Loader for handwritten character dataset

import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path

class HandwrittenDataset:
    """Handwritten Character Dataset loader"""

    def __init__(self, data_path="datasets/Handwritten character dataset"):
        self.data_path = Path(data_path)

    def load_subject_data(self, subject_id=1):
        """Load handwritten data for a specific subject"""
        print(f"📂 Loading handwritten data for subject {subject_id:02d}...")

        mat_file = self.data_path / f"S{subject_id:02d}.mat"

        if not mat_file.exists():
            print(f"❌ File not found: {mat_file}")
            return None, None

        try:
            # Load .mat file
            data = loadmat(str(mat_file))

            # Remove metadata keys
            data_keys = [k for k in data.keys() if not k.startswith('__')]
            print(f"📊 Available keys: {data_keys}")

            # Extract EEG data and labels from nested structure
            eeg_data_list = []
            labels_list = []

            for key in data_keys:
                print(f"\n🔍 Processing key: {key}")
                value = data[key]

                if hasattr(value, 'shape') and value.shape == (1, 1):
                    # Extract nested data
                    nested_data = value[0, 0]

                    if hasattr(nested_data, 'dtype') and nested_data.dtype.names:
                        print(f"  📊 Nested fields: {nested_data.dtype.names}")

                        # Look for EEG data (BrainVisionRDA_data)
                        if 'BrainVisionRDA_data' in nested_data.dtype.names:
                            eeg_array = nested_data['BrainVisionRDA_data'][0]
                            print(f"  ✅ EEG data found: {eeg_array.shape}")
                            print(f"  📊 EEG data type: {type(eeg_array)}")

                            # Only add if it's a proper 2D array (timepoints x channels)
                            if len(eeg_array.shape) == 2 and eeg_array.shape[0] > 100:
                                eeg_data_list.append(eeg_array)
                                print(f"  ✅ Added EEG data: {eeg_array.shape}")
                            else:
                                print(f"  ⚠️ Skipping EEG data with shape: {eeg_array.shape}")

                        # Look for markers/labels
                        if 'EyeblockMarker_data' in nested_data.dtype.names:
                            marker_array = nested_data['EyeblockMarker_data'][0]
                            print(f"  ✅ Markers found: {marker_array.shape}")

                            # Extract unique markers (excluding start/stop markers)
                            unique_markers = marker_array[marker_array != 1000]
                            unique_markers = unique_markers[unique_markers != -1000]

                            if len(unique_markers) > 0:
                                print(f"  📊 Unique markers: {np.unique(unique_markers)}")
                                labels_list.extend(unique_markers)

                        if 'ParadigmMarker_data' in nested_data.dtype.names:
                            paradigm_array = nested_data['ParadigmMarker_data'][0]
                            print(f"  ✅ Paradigm markers found: {paradigm_array.shape}")

                            # Extract paradigm labels
                            unique_paradigm = np.unique(paradigm_array)
                            print(f"  📊 Unique paradigm markers: {unique_paradigm}")

            # Combine data from both rounds
            if len(eeg_data_list) >= 2:
                # Combine EEG data from both rounds
                combined_eeg = np.vstack(eeg_data_list)
                print(f"\n✅ Combined EEG data shape: {combined_eeg.shape}")

                # Create labels based on rounds
                round1_size = eeg_data_list[0].shape[0]
                round2_size = eeg_data_list[1].shape[0]

                # Simple labeling: round 1 = 0, round 2 = 1
                combined_labels = np.concatenate([
                    np.zeros(round1_size, dtype=int),
                    np.ones(round2_size, dtype=int)
                ])

                print(f"✅ Combined labels shape: {combined_labels.shape}")
                print(f"📊 Label distribution: {np.bincount(combined_labels)}")

                return combined_eeg, combined_labels

            elif len(eeg_data_list) == 1:
                eeg_data = eeg_data_list[0]
                print(f"\n✅ Single EEG data shape: {eeg_data.shape}")

                # Create dummy labels
                labels = np.zeros(eeg_data.shape[0], dtype=int)
                print(f"✅ Created dummy labels shape: {labels.shape}")

                return eeg_data, labels

            else:
                print("❌ No EEG data found")
                return None, None

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def visualize_data(self, eeg_data, labels, n_examples=4):
        """Visualize handwritten EEG data"""
        if eeg_data is None or labels is None:
            print("❌ No data to visualize")
            return

        print("📊 Creating visualization...")

        # Get unique labels
        unique_labels = np.unique(labels)
        print(f"📊 Found {len(unique_labels)} unique labels: {unique_labels}")

        # Select examples
        fig, axes = plt.subplots(2, n_examples, figsize=(15, 8))

        for i in range(min(n_examples, len(unique_labels))):
            label = unique_labels[i]
            indices = np.where(labels == label)[0]

            if len(indices) > 0:
                idx = indices[0]  # Take first example

                # Plot EEG data
                if len(eeg_data.shape) == 3:  # [trials, channels, timepoints]
                    trial_data = eeg_data[idx]
                elif len(eeg_data.shape) == 2:  # [trials, features] or [channels, timepoints]
                    if eeg_data.shape[0] == len(labels):  # [trials, features]
                        trial_data = eeg_data[idx].reshape(-1, 1)
                    else:  # [channels, timepoints]
                        trial_data = eeg_data

                # Plot average across channels
                axes[0, i].plot(np.mean(trial_data, axis=0) if trial_data.ndim > 1 else trial_data)
                axes[0, i].set_title(f"Label {label}")
                axes[0, i].set_ylabel("EEG (μV)")
                axes[0, i].grid(True, alpha=0.3)

                # Plot channel data (if available)
                if trial_data.ndim > 1:
                    axes[1, i].imshow(trial_data, aspect='auto', cmap='RdBu_r')
                    axes[1, i].set_title("Channels")
                    axes[1, i].set_ylabel("Channels")
                    axes[1, i].set_xlabel("Time")
                else:
                    axes[1, i].plot(trial_data)
                    axes[1, i].set_title("Signal")
                    axes[1, i].set_xlabel("Time")

        plt.suptitle('Handwritten Character EEG Dataset', fontsize=14)
        plt.tight_layout()
        plt.savefig('handwritten_dataset_visualization.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved as 'handwritten_dataset_visualization.png'")

        plt.close()

def main():
    """Test handwritten dataset loader"""
    dataset = HandwrittenDataset()

    # Load data
    eeg_data, labels = dataset.load_subject_data(1)

    if eeg_data is not None:
        # Visualize
        dataset.visualize_data(eeg_data, labels)

        print(f"\n✅ Handwritten dataset loaded successfully!")
        print(f"📊 EEG data shape: {eeg_data.shape}")
        print(f"📊 Labels shape: {labels.shape if labels is not None else 'None'}")
    else:
        print("❌ Failed to load handwritten dataset")

if __name__ == "__main__":
    main()

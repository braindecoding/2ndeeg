#!/usr/bin/env python3
# check_handwritten_dataset.py - Check handwritten character dataset

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def check_mat_file():
    """Check the .mat file in handwritten dataset"""
    print("🔍 Checking Handwritten Character Dataset")
    print("=" * 50)
    
    mat_file = "datasets/Handwritten character dataset/S01.mat"
    
    if not os.path.exists(mat_file):
        print(f"❌ File not found: {mat_file}")
        return
    
    # Check file size
    file_size = os.path.getsize(mat_file) / (1024**2)  # MB
    print(f"📊 File size: {file_size:.1f} MB")
    
    try:
        # Try to load with scipy
        from scipy.io import loadmat
        
        print("📂 Loading .mat file...")
        data = loadmat(mat_file)
        
        print("✅ File loaded successfully!")
        print(f"📊 Keys in .mat file: {list(data.keys())}")
        
        # Remove metadata keys
        data_keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"📊 Data keys: {data_keys}")
        
        # Examine each data key
        for key in data_keys:
            value = data[key]
            print(f"\n🔍 Key: '{key}'")
            print(f"  Type: {type(value)}")
            print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            print(f"  Dtype: {value.dtype if hasattr(value, 'dtype') else 'N/A'}")
            
            if hasattr(value, 'shape') and len(value.shape) > 0:
                print(f"  Min: {np.min(value) if np.issubdtype(value.dtype, np.number) else 'N/A'}")
                print(f"  Max: {np.max(value) if np.issubdtype(value.dtype, np.number) else 'N/A'}")
                print(f"  Sample values: {value.flat[:5] if value.size > 0 else 'Empty'}")
        
        return data
        
    except ImportError:
        print("❌ scipy not available, trying alternative method...")
        try:
            import h5py
            print("📂 Trying to load as HDF5...")
            
            with h5py.File(mat_file, 'r') as f:
                print("✅ File loaded as HDF5!")
                print(f"📊 Keys: {list(f.keys())}")
                
                for key in f.keys():
                    dataset = f[key]
                    print(f"\n🔍 Key: '{key}'")
                    print(f"  Shape: {dataset.shape}")
                    print(f"  Dtype: {dataset.dtype}")
                    
                    if dataset.size > 0 and dataset.size < 1000000:  # Don't load huge arrays
                        data_array = dataset[:]
                        print(f"  Min: {np.min(data_array) if np.issubdtype(data_array.dtype, np.number) else 'N/A'}")
                        print(f"  Max: {np.max(data_array) if np.issubdtype(data_array.dtype, np.number) else 'N/A'}")
                        print(f"  Sample: {data_array.flat[:5] if data_array.size > 0 else 'Empty'}")
                
        except Exception as e:
            print(f"❌ Error loading with h5py: {str(e)}")
            
    except Exception as e:
        print(f"❌ Error loading .mat file: {str(e)}")

def analyze_handwritten_structure():
    """Analyze the structure of handwritten dataset"""
    print("\n🔍 Analyzing Dataset Structure")
    print("=" * 40)
    
    dataset_dir = Path("datasets/Handwritten character dataset")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    print(f"📁 Dataset directory: {dataset_dir}")
    
    # List all files
    files = list(dataset_dir.glob("*"))
    print(f"📊 Files found: {len(files)}")
    
    for file in files:
        file_size = file.stat().st_size / (1024**2)  # MB
        print(f"  📄 {file.name} ({file_size:.1f} MB)")
    
    # Check if there are more subjects
    mat_files = list(dataset_dir.glob("*.mat"))
    pdf_files = list(dataset_dir.glob("*.pdf"))
    
    print(f"\n📊 Summary:")
    print(f"  .mat files: {len(mat_files)}")
    print(f"  .pdf files: {len(pdf_files)}")
    
    if len(mat_files) == 1:
        print("  ⚠️ Only 1 subject found - limited for cross-subject validation")
    elif len(mat_files) > 1:
        print(f"  ✅ {len(mat_files)} subjects found - good for cross-subject validation")
    else:
        print("  ❌ No .mat files found")

def create_handwritten_loader():
    """Create a basic loader for handwritten dataset"""
    print("\n🔧 Creating handwritten dataset loader...")
    
    loader_code = '''#!/usr/bin/env python3
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
            
            # Try to identify EEG data and labels
            eeg_data = None
            labels = None
            
            for key in data_keys:
                value = data[key]
                print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
                
                # Heuristic to identify EEG data (usually largest array)
                if hasattr(value, 'shape') and len(value.shape) >= 2:
                    if eeg_data is None or value.size > eeg_data.size:
                        eeg_data = value
                        eeg_key = key
                
                # Heuristic to identify labels (usually 1D array with fewer unique values)
                if hasattr(value, 'shape') and len(value.shape) == 1:
                    unique_vals = len(np.unique(value))
                    if unique_vals < 50:  # Likely labels
                        labels = value
                        label_key = key
            
            if eeg_data is not None:
                print(f"✅ EEG data identified: '{eeg_key}' with shape {eeg_data.shape}")
            
            if labels is not None:
                print(f"✅ Labels identified: '{label_key}' with shape {labels.shape}")
                print(f"📊 Unique labels: {np.unique(labels)}")
            
            return eeg_data, labels
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
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
        
        print(f"\\n✅ Handwritten dataset loaded successfully!")
        print(f"📊 EEG data shape: {eeg_data.shape}")
        print(f"📊 Labels shape: {labels.shape if labels is not None else 'None'}")
    else:
        print("❌ Failed to load handwritten dataset")

if __name__ == "__main__":
    main()
'''
    
    with open('handwritten_dataset_loader.py', 'w') as f:
        f.write(loader_code)
    
    print("✅ Created: handwritten_dataset_loader.py")

def main():
    """Main function"""
    print("🔍 Handwritten Character Dataset Analysis")
    print("=" * 50)
    
    # Check dataset structure
    analyze_handwritten_structure()
    
    # Check .mat file
    data = check_mat_file()
    
    # Create loader
    create_handwritten_loader()
    
    print(f"\n📝 Summary:")
    print("✅ Dataset structure analyzed")
    print("✅ .mat file examined")
    print("✅ Loader script created")
    print("\n🚀 Next steps:")
    print("1. Run: python handwritten_dataset_loader.py")
    print("2. Analyze the data structure")
    print("3. Adapt for cross-subject validation")

if __name__ == "__main__":
    main()

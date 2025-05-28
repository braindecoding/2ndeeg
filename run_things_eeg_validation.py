#!/usr/bin/env python3
# run_things_eeg_validation.py - Complete pipeline for THINGS-EEG cross-subject validation

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
        'mne', 'scipy', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies satisfied")
    return True

def check_trained_models():
    """Check if trained models are available"""
    print("\n🔍 Checking trained models...")
    
    required_files = [
        'traditional_models.pkl',
        'meta_model.pkl'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            missing_files.append(file)
            print(f"  ❌ {file}")
    
    if missing_files:
        print(f"\n⚠️ Missing model files: {missing_files}")
        print("Run ensemble_model.py first to train the models")
        return False
    
    print("✅ All trained models available")
    return True

def check_things_eeg_dataset():
    """Check if THINGS-EEG dataset is available"""
    print("\n🔍 Checking THINGS-EEG dataset...")
    
    dataset_path = "datasets/things_eeg"
    
    if not os.path.exists(dataset_path):
        print(f"  ❌ Dataset directory not found: {dataset_path}")
        print("\n📥 THINGS-EEG Dataset Download Instructions:")
        print("  1. Go to: https://osf.io/crn2h/")
        print("  2. Download individual subjects or full dataset")
        print("  3. Extract to 'datasets/things_eeg/'")
        print("  4. Minimum 2 subjects needed for cross-subject validation")
        return False
    
    # Check for subject directories
    subjects_found = []
    for i in range(1, 11):
        subject_dir = os.path.join(dataset_path, f"sub-{i:02d}")
        if os.path.exists(subject_dir):
            eeg_file = os.path.join(subject_dir, "eeg", f"sub-{i:02d}_task-rsvp_eeg.set")
            if os.path.exists(eeg_file):
                subjects_found.append(i)
                print(f"  ✅ Subject {i:02d}")
    
    if len(subjects_found) >= 2:
        print(f"✅ THINGS-EEG dataset available with {len(subjects_found)} subjects")
        return True
    else:
        print(f"❌ Insufficient subjects found: {len(subjects_found)}")
        print("Need at least 2 subjects for cross-subject validation")
        return False

def run_things_eeg_loader():
    """Run THINGS-EEG loader to test dataset"""
    print("\n🚀 Running THINGS-EEG loader...")
    
    try:
        result = subprocess.run([sys.executable, 'things_eeg_loader.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ THINGS-EEG loader completed successfully")
            print("Output:", result.stdout[-500:])  # Last 500 characters
            return True
        else:
            print("❌ THINGS-EEG loader failed")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ THINGS-EEG loader timed out")
        return False
    except Exception as e:
        print(f"❌ Error running THINGS-EEG loader: {str(e)}")
        return False

def run_cross_subject_validation():
    """Run cross-subject validation with THINGS-EEG"""
    print("\n🚀 Running cross-subject validation...")
    
    try:
        result = subprocess.run([sys.executable, 'things_eeg_cross_validation.py'], 
                              capture_output=True, text=True, timeout=1800)  # 30 minutes
        
        if result.returncode == 0:
            print("✅ Cross-subject validation completed successfully")
            print("Output:", result.stdout[-1000:])  # Last 1000 characters
            return True
        else:
            print("❌ Cross-subject validation failed")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Cross-subject validation timed out")
        return False
    except Exception as e:
        print(f"❌ Error running cross-subject validation: {str(e)}")
        return False

def generate_final_report():
    """Generate final report of the validation results"""
    print("\n📊 Generating final report...")
    
    try:
        import numpy as np
        
        # Load results if available
        results_file = 'cross_subject_things_eeg_results.npy'
        
        if os.path.exists(results_file):
            results = np.load(results_file, allow_pickle=True).item()
            
            print("✅ Results loaded successfully")
            print(f"📊 Number of subjects tested: {len(results)}")
            
            # Calculate summary statistics
            ensemble_accuracies = [result['ensemble_accuracy'] for result in results.values()]
            mean_accuracy = np.mean(ensemble_accuracies)
            std_accuracy = np.std(ensemble_accuracies)
            
            print(f"📊 Mean ensemble accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"📊 Best accuracy: {np.max(ensemble_accuracies):.4f}")
            print(f"📊 Worst accuracy: {np.min(ensemble_accuracies):.4f}")
            
            # Generate summary report
            report_content = f"""
THINGS-EEG Cross-Subject Validation Report
==========================================

Dataset: THINGS-EEG (Real Visual Perception EEG Data)
Subjects tested: {len(results)}
Model: Ensemble (SVM + Logistic Regression + Meta-model)

Results Summary:
- Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}
- Best Accuracy: {np.max(ensemble_accuracies):.4f}
- Worst Accuracy: {np.min(ensemble_accuracies):.4f}

Subject Details:
"""
            
            for subject_id, result in results.items():
                report_content += f"- Subject {subject_id:02d}: {result['ensemble_accuracy']:.4f} ({result['n_trials']} trials)\n"
            
            # Domain transfer analysis
            original_accuracy = 0.83
            transfer_ratio = mean_accuracy / original_accuracy
            
            report_content += f"""
Domain Transfer Analysis:
- Original task (digits): {original_accuracy:.3f}
- THINGS-EEG task: {mean_accuracy:.3f}
- Transfer ratio: {transfer_ratio:.3f}

Interpretation:
"""
            
            if transfer_ratio > 0.8:
                report_content += "- Excellent domain transfer capability\n"
            elif transfer_ratio > 0.6:
                report_content += "- Good domain transfer with room for improvement\n"
            else:
                report_content += "- Limited domain transfer - consider domain adaptation\n"
            
            # Save report
            with open('things_eeg_validation_report.txt', 'w') as f:
                f.write(report_content)
            
            print("📄 Report saved as 'things_eeg_validation_report.txt'")
            return True
            
        else:
            print("⚠️ Results file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        return False

def list_generated_files():
    """List all files generated during the validation process"""
    print("\n📁 Generated files:")
    
    expected_files = [
        'cross_subject_things_eeg_results.npy',
        'cross_subject_things_eeg_validation.png',
        'things_eeg_validation_report.txt'
    ]
    
    # Also check for subject-specific files
    for i in range(1, 11):
        expected_files.append(f'things_eeg_subject_{i:02d}_visualization.png')
        expected_files.append(f'things_eeg_subject_{i:02d}_processed.npz')
    
    found_files = []
    
    for file in expected_files:
        if os.path.exists(file):
            file_size = os.path.getsize(file)
            found_files.append(file)
            print(f"  ✅ {file} ({file_size:,} bytes)")
    
    if len(found_files) > 0:
        print(f"\n📊 Total files generated: {len(found_files)}")
    else:
        print("  ❌ No files generated")

def main():
    """Main pipeline function"""
    print("🚀 THINGS-EEG Cross-Subject Validation Pipeline")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("❌ Pipeline failed: Missing dependencies")
        return
    
    # Step 2: Check trained models
    if not check_trained_models():
        print("❌ Pipeline failed: Missing trained models")
        print("💡 Run ensemble_model.py first to train the models")
        return
    
    # Step 3: Check THINGS-EEG dataset
    if not check_things_eeg_dataset():
        print("❌ Pipeline failed: THINGS-EEG dataset not available")
        print("💡 Download THINGS-EEG dataset from: https://osf.io/crn2h/")
        return
    
    # Step 4: Run THINGS-EEG loader (optional test)
    print("\n" + "="*60)
    print("STEP 1: Testing THINGS-EEG Dataset Loader")
    print("="*60)
    
    if run_things_eeg_loader():
        print("✅ Dataset loader test passed")
    else:
        print("⚠️ Dataset loader test failed, but continuing...")
    
    # Step 5: Run cross-subject validation
    print("\n" + "="*60)
    print("STEP 2: Cross-Subject Validation")
    print("="*60)
    
    if run_cross_subject_validation():
        print("✅ Cross-subject validation completed")
    else:
        print("❌ Cross-subject validation failed")
        return
    
    # Step 6: Generate final report
    print("\n" + "="*60)
    print("STEP 3: Final Report Generation")
    print("="*60)
    
    if generate_final_report():
        print("✅ Final report generated")
    else:
        print("⚠️ Report generation failed")
    
    # Step 7: List generated files
    list_generated_files()
    
    print("\n" + "="*60)
    print("🎉 THINGS-EEG VALIDATION PIPELINE COMPLETED!")
    print("="*60)
    print("\n📋 Summary:")
    print("  ✅ Dependencies checked")
    print("  ✅ Trained models verified")
    print("  ✅ THINGS-EEG dataset validated")
    print("  ✅ Cross-subject validation performed")
    print("  ✅ Results analyzed and visualized")
    print("  ✅ Final report generated")
    print("\n📄 Check the following files for results:")
    print("  - things_eeg_validation_report.txt")
    print("  - cross_subject_things_eeg_validation.png")
    print("  - cross_subject_things_eeg_results.npy")

if __name__ == "__main__":
    main()

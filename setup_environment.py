#!/usr/bin/env python3
# setup_environment.py - Environment setup and validation script

import subprocess
import sys
import os
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_python_version():
    """Check Python version compatibility"""
    print_header("PYTHON VERSION CHECK")
    
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ ERROR: Python 3 is required")
        return False
    
    if version.minor < 8:
        print("⚠️  WARNING: Python 3.8+ recommended")
        print("   Current version may work but not fully tested")
    
    print("✅ Python version is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print_header("INSTALLING REQUIREMENTS")
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ ERROR: {requirements_file} not found")
        print("   Make sure you're in the correct directory")
        return False
    
    try:
        print("📦 Installing packages from requirements.txt...")
        print("   This may take a few minutes...")
        
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ], capture_output=True, text=True, check=True)
        
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to install requirements")
        print(f"   Error: {e}")
        print(f"   Output: {e.stdout}")
        print(f"   Error output: {e.stderr}")
        return False

def verify_critical_packages():
    """Verify critical packages are installed with correct versions"""
    print_header("VERIFYING CRITICAL PACKAGES")
    
    critical_packages = {
        'numpy': '1.26.2',
        'scikit-learn': '1.3.2',
        'pandas': '2.2.3',
        'matplotlib': '3.8.2'
    }
    
    all_good = True
    
    for package, expected_version in critical_packages.items():
        try:
            if package == 'scikit-learn':
                import sklearn
                actual_version = sklearn.__version__
                package_name = 'sklearn'
            else:
                module = __import__(package)
                actual_version = module.__version__
                package_name = package
            
            print(f"📦 {package_name}: {actual_version}", end="")
            
            if actual_version == expected_version:
                print(" ✅")
            else:
                print(f" ⚠️  (expected {expected_version})")
                if package == 'scikit-learn' and actual_version.startswith('1.6'):
                    print(f"   ❌ CRITICAL: scikit-learn 1.6.x has compatibility issues!")
                    print(f"   Please downgrade: pip install scikit-learn==1.3.2")
                    all_good = False
                
        except ImportError:
            print(f"❌ {package}: Not installed")
            all_good = False
        except AttributeError:
            print(f"⚠️  {package}: Installed but version unknown")
    
    return all_good

def test_compatibility():
    """Test critical functionality"""
    print_header("TESTING COMPATIBILITY")
    
    try:
        print("🧪 Testing numpy...")
        import numpy as np
        X = np.random.randn(10, 5).astype(np.float64)
        print("   ✅ NumPy working")
        
        print("🧪 Testing scikit-learn...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Test the problematic case
        y = np.random.randint(0, 2, 10).astype(np.int32)
        
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X, y)
        pred = rf.predict(X)
        print("   ✅ RandomForest working")
        
        lr = LogisticRegression(random_state=42)
        lr.fit(X, y)
        pred = lr.predict(X)
        print("   ✅ LogisticRegression working")
        
        print("🧪 Testing pandas...")
        import pandas as pd
        df = pd.DataFrame(X)
        print("   ✅ Pandas working")
        
        print("🧪 Testing matplotlib...")
        import matplotlib.pyplot as plt
        print("   ✅ Matplotlib working")
        
        print("\n✅ All critical packages working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Compatibility test failed: {str(e)}")
        print("\nThis usually indicates a package version conflict.")
        print("Try reinstalling with: pip install -r requirements.txt --force-reinstall")
        return False

def check_data_files():
    """Check if required data files exist"""
    print_header("CHECKING DATA FILES")
    
    required_files = [
        'handwritten_eeg_data.npy',
        'handwritten_session_labels.npy'
    ]
    
    all_files_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024**2)
            print(f"✅ {file}: {size_mb:.1f} MB")
        else:
            print(f"❌ {file}: Not found")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n⚠️  Some data files are missing.")
        print("   Make sure you have extracted the handwritten EEG dataset.")
        print("   Run debug_handwritten_data.py to extract data if needed.")
    
    return all_files_exist

def run_minimal_test():
    """Run the minimal test script"""
    print_header("RUNNING MINIMAL TEST")
    
    if not os.path.exists('minimal_test.py'):
        print("❌ minimal_test.py not found")
        return False
    
    try:
        print("🧪 Running minimal_test.py...")
        result = subprocess.run([
            sys.executable, 'minimal_test.py'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Minimal test passed!")
            print("\nTest output:")
            print("-" * 40)
            print(result.stdout)
            return True
        else:
            print("❌ Minimal test failed!")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Minimal test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running minimal test: {str(e)}")
        return False

def create_virtual_environment_guide():
    """Create a guide for virtual environment setup"""
    print_header("VIRTUAL ENVIRONMENT GUIDE")
    
    guide = """
# Virtual Environment Setup Guide

## Option 1: Using venv (recommended)
python -m venv eeg_analysis_env
source eeg_analysis_env/bin/activate  # On Windows: eeg_analysis_env\\Scripts\\activate
pip install -r requirements.txt

## Option 2: Using conda
conda create -n eeg_analysis python=3.12
conda activate eeg_analysis
pip install -r requirements.txt

## Deactivate environment when done
deactivate  # for venv
conda deactivate  # for conda

## Reactivate environment later
source eeg_analysis_env/bin/activate  # venv
conda activate eeg_analysis  # conda
"""
    
    with open('virtual_environment_guide.txt', 'w') as f:
        f.write(guide)
    
    print("📝 Created virtual_environment_guide.txt")
    print("   Recommended: Use virtual environment for isolation")

def main():
    """Main setup function"""
    print("🔧 Cross-Subject Validation Environment Setup")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Python version incompatible")
        return False
    
    # Step 2: Install requirements
    if not install_requirements():
        print("\n❌ Setup failed: Could not install requirements")
        return False
    
    # Step 3: Verify packages
    if not verify_critical_packages():
        print("\n❌ Setup failed: Package verification failed")
        return False
    
    # Step 4: Test compatibility
    if not test_compatibility():
        print("\n❌ Setup failed: Compatibility test failed")
        return False
    
    # Step 5: Check data files
    data_files_ok = check_data_files()
    
    # Step 6: Run minimal test (if data available)
    if data_files_ok:
        test_ok = run_minimal_test()
    else:
        test_ok = False
        print("\n⚠️  Skipping minimal test (data files missing)")
    
    # Step 7: Create guides
    create_virtual_environment_guide()
    
    # Final summary
    print_header("SETUP SUMMARY")
    
    print(f"✅ Python version: Compatible")
    print(f"✅ Requirements: Installed")
    print(f"✅ Packages: Verified")
    print(f"✅ Compatibility: Tested")
    print(f"{'✅' if data_files_ok else '⚠️ '} Data files: {'Available' if data_files_ok else 'Missing'}")
    print(f"{'✅' if test_ok else '⚠️ '} Minimal test: {'Passed' if test_ok else 'Skipped/Failed'}")
    
    if data_files_ok and test_ok:
        print("\n🎉 Environment setup completed successfully!")
        print("   You can now run: python efficient_cross_validation.py")
    elif not data_files_ok:
        print("\n⚠️  Environment setup completed with warnings.")
        print("   Extract data files first, then run minimal_test.py")
    else:
        print("\n❌ Environment setup completed with errors.")
        print("   Check the error messages above and retry.")
    
    return data_files_ok and test_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

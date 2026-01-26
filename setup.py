#!/usr/bin/env python3
"""
Setup and Authentication Helper
Guides user through Hugging Face authentication
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'ultralytics',
        'huggingface_hub',
        'cv2',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'torch'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_hf_auth():
    """Check Hugging Face authentication"""
    print("\nChecking Hugging Face authentication...")
    
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        
        if token:
            print("✓ Hugging Face authentication found!")
            print(f"  Token: {token[:10]}...{token[-10:]}")
            return True
        else:
            print("✗ Not authenticated with Hugging Face")
            return False
    except Exception as e:
        print(f"✗ Error checking authentication: {e}")
        return False


def authenticate_hf():
    """Guide user through HF authentication"""
    print("\n" + "=" * 70)
    print("HUGGING FACE AUTHENTICATION REQUIRED")
    print("=" * 70)
    print("\nThe VehicleNet-Y26x model requires Hugging Face authentication.")
    print("\nOptions:")
    print("  1. Interactive login (recommended)")
    print("  2. Manual token entry")
    print("  3. Skip (you can authenticate later)")
    
    choice = input("\nChoose option (1/2/3): ").strip()
    
    if choice == '1':
        print("\nRunning: huggingface-cli login")
        print("\nYou'll be prompted to:")
        print("  1. Visit https://huggingface.co/settings/tokens")
        print("  2. Create a new token (or copy existing one)")
        print("  3. Paste it when prompted")
        print("\nPress Enter to continue...")
        input()
        
        os.system('huggingface-cli login')
        
        # Verify
        if check_hf_auth():
            print("\n✓ Authentication successful!")
            return True
        else:
            print("\n⚠ Authentication may have failed. Try again if needed.")
            return False
    
    elif choice == '2':
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        token = input("Paste your token here: ").strip()
        
        if token:
            try:
                from huggingface_hub import HfFolder
                HfFolder.save_token(token)
                print("✓ Token saved!")
                return True
            except Exception as e:
                print(f"✗ Error saving token: {e}")
                return False
        else:
            print("⚠ No token provided")
            return False
    
    else:
        print("\n⚠ Skipping authentication")
        print("You can authenticate later with: huggingface-cli login")
        return False


def verify_project_structure():
    """Verify project files exist"""
    print("\nVerifying project structure...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'config/vehicle_weights.json',
        'src/vehicle_classifier.py',
        'src/density_calculator.py',
        'src/time_series_generator.py',
        'src/visualizer.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✓ All project files present!")
    else:
        print("\n⚠ Some files are missing")
    
    return all_exist


def main():
    print("=" * 70)
    print("TRAFFIC ANALYSIS PIPELINE - SETUP")
    print("=" * 70)
    print()
    
    # Check project structure
    if not verify_project_structure():
        print("\n✗ Project structure incomplete")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n✗ Please install dependencies first")
        sys.exit(1)
    
    # Check/setup authentication
    if not check_hf_auth():
        print("\n⚠ Hugging Face authentication not found")
        
        response = input("\nWould you like to authenticate now? (y/n): ").strip().lower()
        if response == 'y':
            authenticate_hf()
        else:
            print("\n⚠ You'll need to authenticate before running the pipeline")
            print("Run: huggingface-cli login")
    
    # Create data directories
    print("\nCreating data directories...")
    Path('data/input').mkdir(parents=True, exist_ok=True)
    Path('data/output').mkdir(parents=True, exist_ok=True)
    print("  ✓ data/input/")
    print("  ✓ data/output/")
    
    # Final summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Place your video in: data/input/")
    print("  2. Run: python main.py --video data/input/your_video.mp4")
    print("  3. Check results in: data/output/")
    print("\nFor more options, see:")
    print("  - QUICKSTART.md")
    print("  - README.md")
    print("  - python examples.py")
    print("=" * 70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Face Recognition Attendance System
A complete solution for automated attendance tracking using face recognition
"""

import os
import sys
from capture_image import take_images
from train_model import train_model
from recognize_attendance import recognize_attendance
from camera_test import test_camera

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🎯 FACE RECOGNITION ATTENDANCE SYSTEM")
    print("=" * 60)
    print("📚 Automated attendance tracking using AI face recognition")
    print("🔧 Built with OpenCV and Python")
    print("=" * 60)

def print_menu():
    """Print main menu options"""
    print("\n📋 MAIN MENU")
    print("-" * 30)
    print("1. 📸 Capture Face Images (Add New Student)")
    print("2. 🤖 Train Recognition Model")
    print("3. 📋 Take Attendance")
    print("4. 🎥 Test Camera & Face Detection")
    print("5. 📊 View System Status")
    print("6. 🚪 Exit")
    print("-" * 30)

def check_system_status():
    """Check and display system status"""
    print("\n🔍 SYSTEM STATUS CHECK")
    print("-" * 40)
    
    # Check required files
    required_files = [
        ("haarcascade_frontalface_default.xml", "Face detection model"),
        ("trainer.yml", "Trained recognition model")
    ]
    
    for filename, description in required_files:
        status = "✅ Found" if os.path.exists(filename) else "❌ Missing"
        print(f"{description}: {status}")
    
    # Check directories
    required_dirs = [
        ("TrainingImage", "Training images storage"),
        ("StudentDetails", "Student database"),
        ("Attendance", "Attendance records")
    ]
    
    for dirname, description in required_dirs:
        if os.path.exists(dirname):
            count = len([f for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))])
            print(f"{description}: ✅ Found ({count} files)")
        else:
            print(f"{description}: ❌ Missing")
    
    # Training images analysis
    if os.path.exists("TrainingImage"):
        images = [f for f in os.listdir("TrainingImage") if f.endswith('.jpg')]
        if images:
            # Count unique students
            unique_ids = set()
            for img in images:
                try:
                    student_id = img.split('.')[1]
                    unique_ids.add(student_id)
                except:
                    pass
            print(f"📊 Training data: {len(images)} images from {len(unique_ids)} students")
        else:
            print("📊 Training data: No images found")
    
    print("-" * 40)

def setup_directories():
    """Create necessary directories"""
    directories = ['TrainingImage', 'StudentDetails', 'Attendance']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

def main():
    """Main application loop"""
    print_banner()
    setup_directories()
    
    while True:
        print_menu()
        
        try:
            choice = input("\n🎯 Enter your choice (1-6): ").strip()
            
            if choice == '1':
                print("\n📸 CAPTURE FACE IMAGES")
                print("-" * 30)
                print("💡 Instructions:")
                print("   • Ensure good lighting")
                print("   • Look directly at camera")
                print("   • Keep face centered in frame")
                print("   • System will capture 100 images automatically")
                print()
                take_images()
                
            elif choice == '2':
                print("\n🤖 TRAIN RECOGNITION MODEL")
                print("-" * 30)
                print("💡 This process will:")
                print("   • Analyze all captured face images")
                print("   • Create AI recognition model")
                print("   • Save trained model for attendance")
                print()
                train_model()
                
            elif choice == '3':
                print("\n📋 TAKE ATTENDANCE")
                print("-" * 30)
                print("💡 Instructions:")
                print("   • Students should look at camera one by one")
                print("   • System will automatically detect and record")
                print("   • Green text = Successfully recognized")
                print("   • Press 'q' to finish and save attendance")
                print()
                input("Press Enter to start attendance recognition...")
                recognize_attendance()
                
            elif choice == '4':
                print("\n🎥 CAMERA TEST")
                print("-" * 30)
                test_camera()
                
            elif choice == '5':
                check_system_status()
                
            elif choice == '6':
                print("\n👋 Thank you for using Face Recognition Attendance System!")
                print("🎓 Perfect for schools, offices, and organizations")
                sys.exit(0)
                
            else:
                print("\n❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the system.")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or contact support.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
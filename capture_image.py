import csv
import cv2
import os

def is_number(s):
    """Check if a string represents a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['TrainingImage', 'StudentDetails', 'Attendance']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def take_images():
    """Capture face images for training"""
    # Create directories
    create_directories()
    
    # Get user input
    student_id = input("Enter Your ID (numbers only): ")
    name = input("Enter Your Name (letters only): ")
    
    # Validate input
    if not (is_number(student_id) and name.isalpha()):
        if not is_number(student_id):
            print("❌ Error: Please enter a numeric ID")
        if not name.isalpha():
            print("❌ Error: Please enter an alphabetical name")
        return
    
    # Initialize camera and face detector
    try:
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("❌ Error: Could not access camera")
            return
            
        # Load face detector
        cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            print("❌ Error: haarcascade_frontalface_default.xml not found")
            return
            
        detector = cv2.CascadeClassifier(cascade_path)
        
        print(f"📸 Starting image capture for {name} (ID: {student_id})")
        print("👀 Look at the camera and press 'q' to quit early")
        
        sample_count = 0
        max_samples = 100
        
        while True:
            ret, img = cam.read()
            if not ret:
                print("❌ Error: Could not read from camera")
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Save face image
                sample_count += 1
                filename = f"{name}.{student_id}.{sample_count}.jpg"
                filepath = os.path.join("TrainingImage", filename)
                cv2.imwrite(filepath, gray[y:y + h, x:x + w])
                
                # Show progress
                cv2.putText(img, f"Capturing: {sample_count}/{max_samples}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Capture - Press Q to quit', img)
            
            # Check for quit or max samples
            if cv2.waitKey(1) & 0xFF == ord('q') or sample_count >= max_samples:
                break
        
        # Cleanup
        cam.release()
        cv2.destroyAllWindows()
        
        # Save student details to CSV
        save_student_details(student_id, name)
        
        print(f"✅ Successfully captured {sample_count} images for {name}")
        
    except Exception as e:
        print(f"❌ Error during image capture: {e}")

def save_student_details(student_id, name):
    """Save student details to CSV file"""
    csv_path = os.path.join("StudentDetails", "StudentDetails.csv")
    header = ["Id", "Name"]
    row = [student_id, name]
    
    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    try:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(header)
            
            writer.writerow(row)
        
        print(f"📝 Student details saved: ID={student_id}, Name={name}")
        
    except Exception as e:
        print(f"❌ Error saving student details: {e}")

if __name__ == "__main__":
    take_images()
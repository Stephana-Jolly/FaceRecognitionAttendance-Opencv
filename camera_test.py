import cv2

def test_camera():
    """Test camera and face detection functionality"""
    print("🎥 Testing camera and face detection...")
    print("📝 This will help verify your setup is working correctly")
    
    try:
        # Load face cascade
        cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            print("❌ Error: haarcascade_frontalface_default.xml not found")
            return False
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not access camera")
            return False
        
        print("✅ Camera initialized successfully")
        print("👀 Look at the camera - you should see rectangles around detected faces")
        print("🚪 Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read from camera")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.3, 
                minNeighbors=5, 
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show face count
            cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Camera Test - Press Q to quit', frame)
            
            # Quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error during camera test: {e}")
        return False

if __name__ == "__main__":
    test_camera()
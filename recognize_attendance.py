import datetime
import os
import cv2
import pandas as pd

def load_student_data():
    """Load student data from CSV"""
    csv_path = os.path.join("StudentDetails", "StudentDetails.csv")
    
    if not os.path.exists(csv_path):
        print("❌ Error: StudentDetails.csv not found. Please add students first.")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"📚 Loaded {len(df)} students from database")
        return df
    except Exception as e:
        print(f"❌ Error loading student data: {e}")
        return None

def recognize_attendance():
    """Main function for attendance recognition"""
    try:
        # Load trained model
        model_path = "trainer.yml"
        if not os.path.exists(model_path):
            print("❌ Error: trainer.yml not found. Please train the model first.")
            return
        
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(model_path)
        
        # Load face cascade
        cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            print("❌ Error: haarcascade_frontalface_default.xml not found")
            return
        
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load student data
        df = load_student_data()
        if df is None:
            return
        
        # Initialize attendance tracking
        attendance_cols = ['Id', 'Name', 'Date', 'Time']
        attendance = pd.DataFrame(columns=attendance_cols)
        
        # Initialize camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("❌ Error: Could not access camera")
            return
        
        cam.set(3, 640)  # Width
        cam.set(4, 480)  # Height
        
        # Define minimum window size
        min_w = 0.1 * cam.get(3)
        min_h = 0.1 * cam.get(4)
        
        print("🎥 Starting attendance recognition...")
        print("📋 Press 'q' to quit and save attendance")
        print("✅ Green text = Recognized | 🟡 Yellow text = Low confidence | ❌ Red text = Unknown")
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 1.2, 5, 
                minSize=(int(min_w), int(min_h)),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Predict face
                student_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                confidence_score = round(100 - confidence, 2)
                
                # Get student name
                name = "Unknown"
                if confidence_score > 50:  # Confidence threshold
                    try:
                        name_series = df.loc[df['Id'] == student_id]['Name']
                        if not name_series.empty:
                            name = name_series.values[0]
                    except:
                        name = "Unknown"
                
                # Prepare display text
                if confidence_score > 70:
                    display_text = f"{name} [Present]"
                    text_color = (0, 255, 0)  # Green
                    
                    # Record attendance (avoid duplicates)
                    if student_id not in attendance['Id'].values:
                        timestamp = datetime.datetime.now()
                        date_str = timestamp.strftime('%Y-%m-%d')
                        time_str = timestamp.strftime('%H:%M:%S')
                        
                        new_record = pd.DataFrame({
                            'Id': [student_id],
                            'Name': [name],
                            'Date': [date_str],
                            'Time': [time_str]
                        })
                        attendance = pd.concat([attendance, new_record], ignore_index=True)
                        print(f"✅ Attendance recorded: {name} (ID: {student_id})")
                
                elif confidence_score > 50:
                    display_text = f"{name} [Low Confidence]"
                    text_color = (0, 255, 255)  # Yellow
                else:
                    display_text = "Unknown"
                    text_color = (0, 0, 255)  # Red
                
                # Display text on frame
                cv2.putText(frame, display_text, (x + 5, y - 5), font, 0.8, text_color, 2)
                cv2.putText(frame, f"{confidence_score}%", (x + 5, y + h - 5), font, 0.6, text_color, 1)
            
            # Show attendance count
            cv2.putText(frame, f"Present: {len(attendance)}", (10, 30), font, 0.8, (255, 255, 255), 2)
            cv2.imshow('Attendance System - Press Q to quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cam.release()
        cv2.destroyAllWindows()
        
        # Save attendance
        if not attendance.empty:
            save_attendance(attendance)
        else:
            print("📝 No attendance recorded")
    
    except Exception as e:
        print(f"❌ Error during recognition: {e}")

def save_attendance(attendance_df):
    """Save attendance to CSV file"""
    try:
        # Create attendance directory
        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H-%M-%S')
        filename = f"Attendance_{date_str}_{time_str}.csv"
        filepath = os.path.join("Attendance", filename)
        
        # Save to CSV
        attendance_df.to_csv(filepath, index=False)
        
        print(f"✅ Attendance saved successfully!")
        print(f"📁 File: {filepath}")
        print(f"👥 Total attendees: {len(attendance_df)}")
        
        # Display attendance summary
        print("\n📋 Attendance Summary:")
        print("-" * 40)
        for _, row in attendance_df.iterrows():
            print(f"ID: {row['Id']} | Name: {row['Name']} | Time: {row['Time']}")
    
    except Exception as e:
        print(f"❌ Error saving attendance: {e}")

if __name__ == "__main__":
    recognize_attendance()
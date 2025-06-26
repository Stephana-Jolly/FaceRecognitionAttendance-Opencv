import os
import cv2
import numpy as np
from PIL import Image

def get_images_and_labels(path):
    """Extract images and labels from training directory"""
    if not os.path.exists(path):
        print(f"❌ Error: Training directory '{path}' not found")
        return [], []
    
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    
    if not image_paths:
        print("❌ Error: No training images found")
        return [], []
    
    faces = []
    ids = []
    
    print(f"📁 Found {len(image_paths)} training images")
    
    for image_path in image_paths:
        try:
            # Load image and convert to grayscale
            pil_image = Image.open(image_path).convert('L')
            image_np = np.array(pil_image, 'uint8')
            
            # Extract ID from filename (format: name.id.sample.jpg)
            filename = os.path.basename(image_path)
            student_id = int(filename.split(".")[1])
            
            faces.append(image_np)
            ids.append(student_id)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not process {image_path}: {e}")
            continue
    
    print(f"✅ Successfully processed {len(faces)} images")
    return faces, ids

def train_model():
    """Train the face recognition model"""
    try:
        print("🤖 Starting model training...")
        
        # Create recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Check if cascade file exists
        cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            print("❌ Error: haarcascade_frontalface_default.xml not found")
            return
        
        # Get training data
        training_path = "TrainingImage"
        faces, ids = get_images_and_labels(training_path)
        
        if not faces or not ids:
            print("❌ Error: No valid training data found")
            return
        
        # Train the model
        print("🎯 Training model... This may take a few moments")
        recognizer.train(faces, np.array(ids))
        
        # Save the trained model
        model_path = "trainer.yml"
        recognizer.save(model_path)
        
        print(f"✅ Model training completed successfully!")
        print(f"📁 Model saved as: {model_path}")
        print(f"📊 Trained on {len(faces)} images from {len(set(ids))} different people")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    train_model()
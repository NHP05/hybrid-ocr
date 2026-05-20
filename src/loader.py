import os
import cv2

def get_image_paths(input_dir):
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = []
    
    if not os.path.exists(input_dir):
        return []

    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            image_paths.append(os.path.join(input_dir, filename))
            
    return image_paths

def load_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Unreadable/corrupted file skipped: {image_path}")
            return None
            
        # Cleanly load images without aggressive filters
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image_rgb
    except Exception as e:
        print(f"[WARNING] Skipping {image_path} due to error: {e}")
        return None
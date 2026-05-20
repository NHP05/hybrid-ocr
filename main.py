import os
import time
from src.loader import get_image_paths, load_image
from src.model_pipeline import BilingualOCR
from src.exporter import export_results

def main():
    print("=== BATCH BILINGUAL OCR INFERENCE ENGINE ===")
    
    input_dir = "./data/input_images/"
    output_dir = "./data/output_results/"
    os.makedirs(output_dir, exist_ok=True)
    
    print("[INFO] Initializing Hybrid OCR Engine (PaddleOCR + VietOCR)...")
    ocr_model = BilingualOCR()
    
    image_paths = get_image_paths(input_dir)
    print(f"[INFO] Detected {len(image_paths)} images. Starting batch processing...")
    
    for path in image_paths:
        start_time = time.time()
        filename = os.path.basename(path)
        base_filename = os.path.splitext(filename)[0]
        
        image = load_image(path)
        if image is None:
            continue
            
        try:
            extracted_data = ocr_model.process_image(image)
            export_results(extracted_data, output_dir, base_filename)
            
            elapsed_time = round(time.time() - start_time, 2)
            print(f"[SUCCESS] Saved outputs for: {filename} (Time: {elapsed_time}s)")
            
        except Exception as e:
            print(f"[WARNING] Error processing {filename}: {str(e)}")

    print("[INFO] Execution finished cleanly.")

if __name__ == "__main__":
    main()
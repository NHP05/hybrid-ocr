import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import numpy as np
import cv2
from PIL import Image

class BilingualOCR:
    def __init__(self):
        # 1. Initialize PaddleOCR strictly for detection 
        self.det_engine = PaddleOCR(lang='vi')
        
        # 2. Initialize vietocr for high-fidelity recognition
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        self.rec_engine = Predictor(config)
    
    def sort_boxes(self, dt_boxes):
        if not dt_boxes:
            return dt_boxes
        
        # Calculate centers and heights
        centers = []
        heights = []
        for box in dt_boxes:
            pts = np.array(box)
            y_min = np.min(pts[:, 1])
            y_max = np.max(pts[:, 1])
            x_min = np.min(pts[:, 0])
            centers.append([x_min, (y_min + y_max) / 2.0, y_min])
            heights.append(y_max - y_min)
            
        centers = np.array(centers)
        avg_height = np.mean(heights) if len(heights) > 0 else 10
        
        # Sort by y center first
        sorted_indices = np.argsort(centers[:, 1])
        sorted_boxes = [dt_boxes[i] for i in sorted_indices]
        sorted_centers = centers[sorted_indices]
        
        # Group into lines
        lines = []
        current_line = []
        current_y_center = sorted_centers[0][1]
        
        for i, box in enumerate(sorted_boxes):
            if abs(sorted_centers[i][1] - current_y_center) < 0.7 * avg_height:
                current_line.append((box, sorted_centers[i]))
            else:
                lines.append(current_line)
                current_line = [(box, sorted_centers[i])]
                current_y_center = sorted_centers[i][1]
                
        if current_line:
            lines.append(current_line)
            
        # Sort each line by x
        result_boxes = []
        for line in lines:
            line_sorted = sorted(line, key=lambda x: x[1][0])
            for item in line_sorted:
                result_boxes.append(item[0])
                
        return result_boxes

    def get_crop(self, image, box):
        pts = np.array(box, dtype=np.float32)
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        
        # Ensure boundaries
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        crop = image[y:y+h, x:x+w]
        return crop

    def process_image(self, image):
        # Use PaddleOCR solely for drawing bounding boxes
        result = self.det_engine.ocr(image)
        if not result or result[0] is None:
            return []
            
        boxes = [line[0] for line in result[0]]
        
        # Sort the bounding boxes mathematically from Top-to-Bottom and Left-to-Right
        sorted_boxes = self.sort_boxes(boxes)
        
        extracted_data = []
        for i, box in enumerate(sorted_boxes):
            crop_img = self.get_crop(image, box)
            if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue
                
            pil_img = Image.fromarray(crop_img)
            
            # Predict text and probability natively via VietOCR
            try:
                text, prob = self.rec_engine.predict(pil_img, return_prob=True)
            except AttributeError:
                # Some predictor versions might throw if return_prob fails
                text = self.rec_engine.predict(pil_img)
                prob = 0.99
            except TypeError:
                text = self.rec_engine.predict(pil_img)
                prob = 0.99
                
            coords = np.array(box)
            y_min = np.min(coords[:, 1])

            extracted_data.append({
                'Block_ID': i + 1,
                'Extracted_Text': text,
                'Confidence': round(prob, 4) if isinstance(prob, float) else 0.99,
                'y_min': y_min
            })
            
        return extracted_data
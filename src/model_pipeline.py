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
        # Cấu hình quét nâng cao nhằm thu giữ tối đa các nét chữ mảnh
        self.det_engine = PaddleOCR(
            use_angle_cls=True, 
            lang='vi',
            det_db_thresh=0.15,         # Hạ ngưỡng quét cạnh để bắt trọn khối chữ mờ cuối ảnh
            det_db_box_thresh=0.35,
            det_limit_side_len=4000,    # Nới rộng kích thước xử lý ảnh độ phân giải cao
            show_log=False
        )
        
        config = Cfg.load_config_from_name('vgg_transformer')
        config['device'] = 'cpu'
        config['predictor']['beamsearch'] = False
        self.rec_engine = Predictor(config)
    
    def order_points(self, pts):
        """ Sắp xếp các điểm theo thứ tự hình học cố định: Top-Left, Top-Right, Bottom-Right, Bottom-Left """
        pts = np.array(pts, dtype="float32").reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def perspective_crop(self, img, points):
        """ Làm thẳng dòng chữ bị nghiêng bằng biến đổi phối cảnh và thêm lề an toàn """
        try:
            rect = self.order_points(points)
            (tl, tr, br, bl) = rect
            
            width_top = np.linalg.norm(br - bl)
            width_bottom = np.linalg.norm(tr - tl)
            max_w = int(max(width_top, width_bottom))
            
            height_left = np.linalg.norm(tr - br)
            height_right = np.linalg.norm(tl - bl)
            max_h = int(max(height_left, height_right))
            
            if max_w == 0 or max_h == 0:
                return np.array([])
                
            dst_pts = np.float32([[0, 0], [max_w, 0], [max_w, max_h], [0, max_h]])
            M = cv2.getPerspectiveTransform(rect, dst_pts)
            warped = cv2.warpPerspective(img, M, (max_w, max_h), borderMode=cv2.BORDER_REPLICATE)
            
            # Thêm viền trắng rìa chữ để tránh lỗi mất dấu của VietOCR
            padded_chip = cv2.copyMakeBorder(warped, 6, 6, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            return padded_chip
        except Exception:
            return np.array([])

    def sort_boxes(self, dt_boxes):
        # SỬA LỖI TRỰC TIẾP: Kiểm tra rỗng bằng len() thay vì check toán tử logic mảng NumPy
        if dt_boxes is None or len(dt_boxes) == 0:
            return []
        
        # SỬA LỖI SẬP: Bọc enumerate để lấy index x[0] làm mốc phá vỡ liên kết trùng tọa độ Y
        indexed_boxes = list(enumerate(dt_boxes))
        
        def get_y_center(item):
            idx, box = item
            pts = np.array(box)
            return (np.min(pts[:, 1]) + np.max(pts[:, 1])) / 2.0
            
        indexed_boxes = sorted(indexed_boxes, key=lambda x: (get_y_center(x), x[0]))
        dt_boxes = [x[1] for x in indexed_boxes]
        
        lines = []
        current_line = []
        
        if len(dt_boxes) > 0:
            first_box = np.array(dt_boxes[0])
            prev_y_center = (np.min(first_box[:, 1]) + np.max(first_box[:, 1])) / 2.0
            prev_h = np.max(first_box[:, 1]) - np.min(first_box[:, 1])
            
            for box in dt_boxes:
                pts = np.array(box)
                y_center = (np.min(pts[:, 1]) + np.max(pts[:, 1])) / 2.0
                box_height = np.max(pts[:, 1]) - np.min(pts[:, 1])
                
                if abs(y_center - prev_y_center) < (max(prev_h, box_height) * 0.45):
                    current_line.append(box)
                else:
                    # SỬA LỖI SẬP TIÊU CHÍ 2: Sắp xếp hàng ngang lề trái X sử dụng bộ index an toàn Tuple
                    indexed_curr = list(enumerate(current_line))
                    indexed_curr = sorted(indexed_curr, key=lambda x: (np.min(np.array(x[1])[:, 0]), x[0]))
                    current_line = [x[1] for x in indexed_curr]
                    
                    lines.extend(current_line)
                    current_line = [box]
                    prev_y_center = y_center
                    prev_h = box_height
                    
            if len(current_line) > 0:
                indexed_curr = list(enumerate(current_line))
                indexed_curr = sorted(indexed_curr, key=lambda x: (np.min(np.array(x[1])[:, 0]), x[0]))
                current_line = [x[1] for x in indexed_curr]
                lines.extend(current_line)
                
        return lines

    def process_image(self, image):
        dt_boxes, _ = self.det_engine.text_detector(image)
        
        # SỬA LỖI ĐIỀU KIỆN LOGIC: Kiểm tra độ dài an toàn cho mảng NumPy đầu ra của Paddle
        if dt_boxes is None or len(dt_boxes) == 0:
            return []
            
        sorted_boxes = self.sort_boxes(dt_boxes)
        extracted_data = []
        
        for i, box in enumerate(sorted_boxes):
            crop_img = self.perspective_crop(image, box)
            if crop_img is None or getattr(crop_img, 'size', 0) == 0:
                continue
            pil_img = Image.fromarray(crop_img)
            try:
                text = self.rec_engine.predict(pil_img)
            except Exception:
                text = ""
                
            coords = np.array(box)
            y_min = np.min(coords[:, 1])

            extracted_data.append({
                'Block_ID': i + 1,
                'Extracted_Text': text.strip(),
                'y_min': y_min
            })
        return extracted_data

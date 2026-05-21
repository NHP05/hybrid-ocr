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
            print(f"[WARNING] Không thể đọc file: {image_path}")
            return None
            
        # 1. BẢO VỆ BIÊN: Thêm viền trắng 30px xung quanh TOÀN BỘ bức ảnh gốc
        # Giúp thuật toán nhận diện không bỏ sót các dòng chữ kết luận/chữ ký nằm sát mép ảnh
        image_padded = cv2.copyMakeBorder(image, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # 2. Phóng đại hình ảnh 2x bằng thuật toán nội suy Cubic để làm rõ nét chữ in kim nhỏ mờ
        h, w = image_padded.shape[:2]
        image_resized = cv2.resize(image_padded, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # 3. Chuyển sang hệ màu RGB chuẩn hóa cung cấp cho mô hình deep learning
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        return image_rgb
    except Exception as e:
        print(f"[WARNING] Lỗi xử lý loader cho {image_path}: {e}")
        return None
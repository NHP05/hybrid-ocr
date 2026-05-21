import os

def export_results(extracted_data, output_dir, base_filename):
    # Chỉ định đường dẫn xuất file .txt thuần túy
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    
    # Nếu không có dữ liệu trích xuất, tạo file văn bản trống
    if not extracted_data:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("")
        return

    paragraphs = []
    current_paragraph_lines = []
    
    # Lấy tọa độ Y dòng đầu tiên làm mốc
    prev_y = extracted_data[0]['y_min']
    
    for i, item in enumerate(extracted_data):
        text = item['Extracted_Text']
        current_y = item['y_min']
        
        # Nếu khoảng cách dòng chiều dọc > 20 pixel, gom cụm cũ thành 1 đoạn và xuống dòng trống
        if i > 0 and (current_y - prev_y > 20):
            if current_paragraph_lines:
                paragraphs.append(" ".join(current_paragraph_lines))
                current_paragraph_lines = []
        
        current_paragraph_lines.append(text)
        prev_y = current_y
        
    # Gom đoạn văn bản cuối cùng còn sót lại trong vòng lặp
    if current_paragraph_lines:
        paragraphs.append(" ".join(current_paragraph_lines))
        
    # Nối các đoạn văn bản lại với nhau, phân tách bằng 2 dấu xuống dòng (\n\n) chuẩn tệp mẫu
    full_plain_text = "\n\n".join(paragraphs)
    
    # Ghi file chuẩn mã hóa quốc tế UTF-8
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_plain_text)
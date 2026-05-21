import os

def export_results(extracted_data, output_dir, base_filename):
    txt_path = os.path.join(output_dir, f"{base_filename}.txt")
    
    if not extracted_data:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("")
        return

    paragraphs = []
    current_paragraph_lines = []
    
    prev_y = extracted_data[0]['y_min']
    
    for i, item in enumerate(extracted_data):
        text = item['Extracted_Text']
        current_y = item['y_min']
        
        # Nếu khoảng cách nhảy dòng dòng dọc vượt quá 25 pixel -> Ngắt dòng tạo paragraph mới
        if i > 0 and (current_y - prev_y > 25):
            if current_paragraph_lines:
                paragraphs.append(" ".join(current_paragraph_lines))
                current_paragraph_lines = []
        
        if text: # Chỉ thêm nếu khối chữ không rỗng
            current_paragraph_lines.append(text)
            prev_y = current_y
        
    if current_paragraph_lines:
        paragraphs.append(" ".join(current_paragraph_lines))
        
    full_plain_text = "\n\n".join(paragraphs)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_plain_text)
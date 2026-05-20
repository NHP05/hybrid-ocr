# Bilingual Hybrid OCR Pipeline (Vietnamese & English)

## Overview

This project implements a **pure batch-inference hybrid OCR pipeline** specifically tailored for high-accuracy **Vietnamese** and **English** text extraction. 

Many standard OCR solutions (like the default PaddleOCR pipeline) may struggle with precise Vietnamese diacritics or inadvertently corrupt Unicode characters (mojibake) due to internal C++ fallback behaviors or ONEDNN incompatibilities under Linux. To solve this, our pipeline splits the task into two highly specialized steps:

1. **Text Detection (PaddleOCR):** We solely rely on PaddleOCR's highly robust detection models to accurately identify text regions and draw bounding boxes.
2. **Text Recognition (VietOCR):** We process the cropped text segments (chips) using `vietocr` (`vgg_transformer` model) to guarantee 100% preservation of Vietnamese diacritics and precise sequence recognition.

## Features

- **Hybrid Engine**: PaddleOCR for localization + VietOCR for sequence recognition.
- **Automated Batch Processing**: Scans input directories dynamically and processes valid images (`.jpg`, `.png`, `.webp`, etc.).
- **Smart Layout Sorting**: Top-to-Bottom, Left-to-Right bounding box sorting algorithm ensures natural reading flow.
- **Rich Data Exports**:
  - Automatically exports raw line-by-line bounding boxes and texts into `.csv` (UTF-8 formatted).
  - Intelligently merges lines based on vertical gap spacing into standard paragraphs inside `.docx` reports.
- **Mojibake Protected**: By strictly bypassing PaddleOCR's recognition phase, we ensure that character encoding remains fully preserved up into final Python outputs.

## Requirements

The codebase explicitly uses stable library versions to bypass known Paddle API issues on `Linux/WSL`. Make sure to install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
*(Key dependencies: `paddlepaddle==2.6.2`, `paddleocr==2.7.3`, `numpy==1.26.4`, `vietocr`)*

## Project Structure

```
├── data/
│   ├── input_images/     # Put all your images to be processed here
│   └── output_results/   # Automatically generated .csv and .docx outputs
├── src/
│   ├── exporter.py               # Exporter engine for CSV/DOCX conversions
│   ├── loader.py                 # File I/O loader and basic filter utilities
│   └── model_pipeline.py         # Core Hybrid OCR Engine class
├── main.py               # Main batch processor runtime script
├── requirements.txt      # Stable pinned dependencies
└── README.md
```

## Usage

1. **Place Input Data**: Drop your images into the `data/input_images/` directory.
2. **Execute Pipeline**: 
   ```bash
   python main.py
   ```
3. **Retrieve Results**: Extracted texts and reports will be mapped iteratively to `data/output_results/` bearing the same original filenames.

## Hardware Support

By default, the current configuration is highly optimized for CPU operations, specifically to bypass hardware API issues often found on bare-metal WSL/Linux executions. `vietocr` uses CPU, and PaddleOCR utilizes CPU configurations without enabling experimental MKLDNN logic, ensuring widespread reliability.
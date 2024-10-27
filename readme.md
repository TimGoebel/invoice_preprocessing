# PDF to Image Processor

A Python-based tool for processing PDF files, converting them into images, and applying various preprocessing techniques, including noise reduction, skew correction, font thinning, line removal, and resizing. Ideal for document processing tasks, such as data extraction and OCR preparation.

## Features

- **Convert PDF to Images**: Converts each page of a PDF file into an image.
- **Image Preprocessing**: Includes noise removal, skew correction, font thinning, and removal of horizontal/vertical lines.
- **Flexible Scaling**: Automatically scales images to specified dimensions.
- **Border Addition**: Adds borders around images to improve readability in processed outputs.

## Requirements

Ensure you have the following Python packages installed:

- `opencv-python`
- `numpy`
- `pymupdf`

You can install the necessary packages using the command:

```bash
git clone https://github.com/TimGoebel/pdf-to-image-processor.git
cd pdf-to-image-processor

└── preprocessing
    ├── INPUT_DATA    # Folder for PDF files
    └── OUTPUT_DATA   # Folder where processed images will be saved

python pdf_to_image_processor.py

from pdf_to_image_processor import PDFImageProcessor

# Define directory paths
data_path = r"C:\path\to\invoices"
processor = PDFImageProcessor(data_path)

# Process PDFs
processor.process_pdfs()

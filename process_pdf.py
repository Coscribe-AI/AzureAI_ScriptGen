import json
import fitz  # PyMuPDF
import PyPDF2
import pytesseract
from PIL import Image
import io

# Configure Tesseract executable path if it's not in PATH already
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'

def pdf_is_digitally_created(pdf_path):
    """Check if a PDF is digitally created or scanned image."""
    document = fitz.open(pdf_path)
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        if page.get_text("text"):
            document.close()
            return True
    document.close()
    return False

def extract_text_from_pdf(pdf_file):
    """Extract text from digital PDF."""
    file_reader = PyPDF2.PdfReader(pdf_file)
    text = []
    for page in file_reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text.append(extracted_text)
    return text

def perform_ocr_on_pdf(pdf_path):
    """Perform OCR on scanned/image-based PDF."""
    document = fitz.open(pdf_path)
    all_text = []
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        text = pytesseract.image_to_string(img)
        all_text.append(text)
    document.close()
    return all_text

def generate_json(text):
    """Generate JSON string from extracted text."""
    return json.dumps({"result": text})

def transcode_paperwork(pdf_path):
    if pdf_is_digitally_created(pdf_path):
        # digital PDF processing
        text = extract_text_from_pdf(pdf_path)
    else:
        # OCR processing for scanned PDFs
        text = perform_ocr_on_pdf(pdf_path)
    
    return generate_json(text)

# Example usage
if __name__ == "__main__":
    pdf_path = 'uploads/brief0.pdf'
    json_string = transcode_paperwork(pdf_path)
    print(json_string)  # You can now use the JSON string directly in your application.
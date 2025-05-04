import os
import pdfplumber
from PyPDF2 import PdfReader
from utils.logger import setup_logger

from gpu_utils import GPUVerifier

# Initialize GPU verification
gpu_verifier = GPUVerifier(require_gpu=True)

logger = setup_logger("PDFUtils")

def pdf_to_text(pdf_path, output_dir=None):
    """
    Convert a PDF file to text.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the text file (if None, saves in the same directory)
    
    Returns:
        Path to the created text file
    """


    logger.info(f"Converting PDF to text: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Determine output directory and filename
    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    filename = os.path.basename(pdf_path)
    base_filename = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{base_filename}.txt")
    
    # Extract text from PDF using both methods for best results
    try:
        # First try pdfplumber which often has better layout recognition
        with pdfplumber.open(pdf_path) as pdf:
            text_content = ""
            for page in pdf.pages:
                text_content += page.extract_text() or ""
                text_content += "\n\n"  # Add page breaks
        
        # If pdfplumber didn't extract any text, fallback to PyPDF2
        if not text_content.strip():
            logger.info("pdfplumber extracted no text, falling back to PyPDF2")
            reader = PdfReader(pdf_path)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() or ""
                text_content += "\n\n"  # Add page breaks
        
        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        logger.info(f"Successfully converted PDF to text: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error converting PDF to text: {str(e)}")
        raise
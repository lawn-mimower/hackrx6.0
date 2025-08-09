import os
from pathlib import Path
import fitz  # PyMuPDF
import pdfkit
from docx import Document
import extract_msg
import email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import logging
import tempfile

logger = logging.getLogger(__name__)

class DocumentConverter:
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._handle_pdf,
            '.docx': self._convert_docx_to_pdf,
            '.doc': self._convert_doc_to_pdf,
            '.eml': self._convert_eml_to_pdf,
            '.msg': self._convert_msg_to_pdf,
            '.html': self._convert_html_to_pdf,
            '.htm': self._convert_html_to_pdf,
            '.txt': self._convert_txt_to_pdf
        }
    
    async def ensure_pdf(self, file_path: Path) -> Path:
        """
        Ensure the document is in PDF format, convert if necessary.
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return file_path
        
        if suffix in self.supported_formats:
            logger.info(f"Converting {suffix} to PDF...")
            return await self.supported_formats[suffix](file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    async def _handle_pdf(self, file_path: Path) -> Path:
        """PDF files need no conversion."""
        return file_path
    
    async def _convert_docx_to_pdf(self, file_path: Path) -> Path:
        """Convert DOCX to PDF using python-docx and PyMuPDF."""
        pdf_path = file_path.with_suffix('.pdf')
        
        # Read DOCX
        doc = Document(file_path)
        
        # Create PDF using PyMuPDF
        pdf_doc = fitz.open()
        page = pdf_doc.new_page()
        
        # Extract text and add to PDF
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        page.insert_text((72, 72), text, fontsize=11)
        
        pdf_doc.save(pdf_path)
        pdf_doc.close()
        
        return pdf_path
    
    async def _convert_doc_to_pdf(self, file_path: Path) -> Path:
        """Convert DOC to PDF (similar to DOCX)."""
        return await self._convert_docx_to_pdf(file_path)
    
    async def _convert_eml_to_pdf(self, file_path: Path) -> Path:
        """Convert EML email to PDF."""
        pdf_path = file_path.with_suffix('.pdf')
        
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        
        # Extract email content
        content = f"From: {msg['from']}\n"
        content += f"To: {msg['to']}\n"
        content += f"Subject: {msg['subject']}\n"
        content += f"Date: {msg['date']}\n\n"
        
        # Get email body
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        else:
            content += msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        # Create PDF
        pdf_doc = fitz.open()
        page = pdf_doc.new_page()
        page.insert_text((72, 72), content, fontsize=11)
        pdf_doc.save(pdf_path)
        pdf_doc.close()
        
        return pdf_path
    
    async def _convert_msg_to_pdf(self, file_path: Path) -> Path:
        """Convert MSG (Outlook) to PDF."""
        pdf_path = file_path.with_suffix('.pdf')
        
        msg = extract_msg.Message(str(file_path))
        
        content = f"From: {msg.sender}\n"
        content += f"To: {msg.to}\n"
        content += f"Subject: {msg.subject}\n"
        content += f"Date: {msg.date}\n\n"
        content += msg.body
        
        # Create PDF
        pdf_doc = fitz.open()
        page = pdf_doc.new_page()
        page.insert_text((72, 72), content, fontsize=11)
        pdf_doc.save(pdf_path)
        pdf_doc.close()
        
        msg.close()
        return pdf_path
    
    async def _convert_html_to_pdf(self, file_path: Path) -> Path:
        """Convert HTML to PDF using pdfkit."""
        pdf_path = file_path.with_suffix('.pdf')
        
        # Use pdfkit for HTML to PDF conversion
        pdfkit.from_file(str(file_path), str(pdf_path))
        
        return pdf_path
    
    async def _convert_txt_to_pdf(self, file_path: Path) -> Path:
        """Convert plain text to PDF."""
        pdf_path = file_path.with_suffix('.pdf')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create PDF
        pdf_doc = fitz.open()
        page = pdf_doc.new_page()
        page.insert_text((72, 72), text, fontsize=11)
        pdf_doc.save(pdf_path)
        pdf_doc.close()
        
        return pdf_path
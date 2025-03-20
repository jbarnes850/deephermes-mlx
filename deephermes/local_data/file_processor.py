"""
File processor for local data integration.

This module provides functionality for processing various file types
and extracting their content for use in RAG and fine-tuning.
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import json
import csv

class FileProcessor:
    """Process local files for RAG and fine-tuning.
    
    This class handles the extraction of content from various file types,
    including text, markdown, PDF, DOCX, JSON, and CSV files.
    """
    
    supported_extensions = {
        '.txt': 'text',
        '.md': 'text',
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.json': 'json',
        '.csv': 'csv'
    }
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize file processor with optional base directory.
        
        Args:
            base_dir: Optional base directory for relative path resolution
        """
        self.base_dir = Path(base_dir) if base_dir else None
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a single file and extract its content with metadata.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary containing file content and metadata
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file type is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        extension = file_path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        file_type = self.supported_extensions[extension]
        processor = getattr(self, f"_process_{file_type}")
        
        content = processor(file_path)
        
        return {
            "filename": file_path.name,
            "path": str(file_path),
            "type": file_type,
            "content": content,
            "metadata": {
                "created": os.path.getctime(file_path),
                "modified": os.path.getmtime(file_path),
                "size": os.path.getsize(file_path)
            }
        }
    
    def process_directory(self, 
                         directory: Union[str, Path], 
                         recursive: bool = True) -> List[Dict[str, Any]]:
        """Process all supported files in a directory.
        
        Args:
            directory: Directory to process
            recursive: Whether to process subdirectories recursively
            
        Returns:
            List of dictionaries containing file content and metadata
            
        Raises:
            ValueError: If the directory does not exist
        """
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"{directory} is not a valid directory")
        
        results = []
        
        pattern = "**/*" if recursive else "*"
        for ext in self.supported_extensions:
            for file_path in directory.glob(f"{pattern}{ext}"):
                if file_path.is_file():
                    try:
                        results.append(self.process_file(file_path))
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return results
    
    def _process_text(self, file_path: Path) -> str:
        """Process text files (txt, md).
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content of the file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _process_pdf(self, file_path: Path) -> str:
        """Process PDF files.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
            
        Note:
            Requires PyPDF2 to be installed
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required for processing PDF files. Install it with 'pip install PyPDF2'.")
        
        text = ""
        with open(file_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _process_docx(self, file_path: Path) -> str:
        """Process DOCX files.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text from the DOCX
            
        Note:
            Requires python-docx to be installed
        """
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx is required for processing DOCX files. Install it with 'pip install python-docx'.")
        
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _process_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process JSON files.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON content
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _process_csv(self, file_path: Path) -> List[List[str]]:
        """Process CSV files.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of rows from the CSV
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return list(reader)

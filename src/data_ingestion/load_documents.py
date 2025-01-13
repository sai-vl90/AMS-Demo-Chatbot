import os
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf

RAW_DATA_DIR = "data/raw"

def load_raw_documents():
    docs = []
    for filename in os.listdir(RAW_DATA_DIR):
        # Construct the full path
        filepath = os.path.join(RAW_DATA_DIR, filename)
        
        # Check for Word docs
        if filename.lower().endswith(".docx"):
            elements = partition_docx(filename=filepath)
            docs.append((filename, elements))
        
        # Check for PDFs
        elif filename.lower().endswith(".pdf"):
            elements = partition_pdf(filename=filepath)
            docs.append((filename, elements))
        
        # Ignore other file types
        else:
            pass

    return docs

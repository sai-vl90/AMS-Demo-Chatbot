import os
from unstructured.partition.docx import partition_docx

RAW_DATA_DIR = "data/raw"

def load_raw_documents():
    docs = []
    for filename in os.listdir(RAW_DATA_DIR):
        if filename.endswith(".docx"):
            filepath = os.path.join(RAW_DATA_DIR, filename)
            # Partition the docx into text elements (unstructured returns a list of elements)
            elements = partition_docx(filename=filepath)
            docs.append((filename, elements))
    return docs

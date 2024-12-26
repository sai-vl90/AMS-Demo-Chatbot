import os
import json
from typing import List, Dict, Tuple
from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings

def load_preprocessed_data(preprocessed_dir: str) -> List[Dict]:
    """
    Load all processed documents from the preprocessed directory.

    Args:
        preprocessed_dir (str): Path to the directory containing preprocessed JSON files.

    Returns:
        List[Dict]: A list of dictionaries representing each preprocessed document.
    """
    docs_data = []
    for filename in os.listdir(preprocessed_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(preprocessed_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    docs_data.append(data)
                print(f"Loaded preprocessed file: {filepath}")
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
    return docs_data

def extract_texts_and_metadata(docs_data: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Extract text chunks and their metadata from processed documents.

    This function processes both text sections and tables, assigning appropriate metadata.

    Args:
        docs_data (List[Dict]): List of preprocessed document dictionaries.

    Returns:
        Tuple[List[str], List[Dict]]: A tuple containing:
            - List of text chunks.
            - List of metadata dictionaries corresponding to each text chunk.
    """
    texts = []
    metadatas = []

    for doc_index, doc in enumerate(docs_data, start=1):
        processed_doc = doc.get('processed_doc', {})
        sections = processed_doc.get('sections', [])
        tables = processed_doc.get('tables', [])
        table_contexts = processed_doc.get('table_contexts', [])
        document_name = processed_doc.get('document_name', f'Unknown_Document_{doc_index}')

        # Process text sections
        for section in sections:
            content = section.get('content', '').strip()
            if not content:
                continue  # Skip empty sections

            title = section.get('title', 'No Title')
            keywords = section.get('keywords', 'No Keywords')
            page_number = section.get('page_number', 'N/A')  # Currently 'N/A'

            metadata = {
                'keywords': keywords,
                'section_title': title,
                'section_level': section.get('level', 1),
                'content_type': 'text',
                'chunk_length': len(content),
                'page_number': page_number,
                'document_name': document_name
            }

            texts.append(content)
            metadatas.append(metadata)

        # Process tables
        for table_index, table in enumerate(tables, start=1):
            table_content = table.strip() if table else 'No Table Content'
            context = table_contexts[table_index - 1] if table_index - 1 < len(table_contexts) else 'No context available.'

            metadata = {
                'keywords': 'No Keywords',
                'section_title': 'Table',
                'section_level': 1,  # Assuming tables are top-level; adjust if necessary
                'content_type': 'table',
                'chunk_length': len(table_content),
                'page_number': 'N/A',  # Placeholder; see limitations
                'document_name': document_name,
                'context': context
            }

            texts.append(table_content)
            metadatas.append(metadata)

    print(f"Extracted {len(texts)} text chunks and {len(metadatas)} metadata entries.")
    return texts, metadatas

def build_deeplake_vectorstore(texts: List[str], metadatas: List[Dict], dataset_path: str):
    """
    Create or update a Deep Lake vector store with provided texts and metadata.

    Args:
        texts (List[str]): List of text chunks to be embedded.
        metadatas (List[Dict]): List of metadata dictionaries corresponding to each text chunk.
        dataset_path (str): The Deep Lake dataset path (e.g., "hub://username/dataset_name").
    """
    # Initialize the HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create the Deep Lake vector store
    print("Building the Deep Lake vector store...")
    db = DeepLake.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        dataset_path=dataset_path,
        overwrite=True  # Set to True to overwrite existing dataset; set to False to append
    )
    print("Deep Lake vector store built successfully!")

def main():
    """
    Main function to orchestrate the loading, extraction, and vector store building processes.
    """
    PREPROCESSED_DIR = "data/preprocessed"

    # Load configuration and set environment variables
    # Assuming you have a config_loader module as in your original script
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(project_root)
    try:
        from src.config_loader import load_config, set_environment_variables
    except ImportError as e:
        print(f"Failed to import config_loader: {e}")
        return

    config = load_config()
    set_environment_variables(config)

    # Retrieve dataset_path from config
    dataset_path = config.get('deeplake', {}).get('dataset_path')
    if not dataset_path:
        print("Error: 'dataset_path' not found in the configuration.")
        return

    # Load preprocessed data
    docs_data = load_preprocessed_data(PREPROCESSED_DIR)
    if not docs_data:
        print("No preprocessed data found. Please ensure the 'data/preprocessed' directory contains JSON files.")
        return

    # Extract texts and metadata
    texts, metadatas = extract_texts_and_metadata(docs_data)

    # Build the Deep Lake vector store
    build_deeplake_vectorstore(texts, metadatas, dataset_path)

if __name__ == "__main__":
    main()

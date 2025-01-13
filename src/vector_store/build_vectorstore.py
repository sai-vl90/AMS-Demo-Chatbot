import os
import json
from typing import List, Dict, Tuple
from langchain_community.vectorstores import AzureSearch
from langchain_huggingface import HuggingFaceEmbeddings
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

def load_preprocessed_data(preprocessed_dir: str) -> List[Dict]:
    """
    Load all processed documents from the preprocessed directory.
    """
    docs_data = []
    print(f"\nLoading all documents from {preprocessed_dir}")
    
    for filename in os.listdir(preprocessed_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(preprocessed_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    docs_data.append(data)
                    print(f"Loaded file: {filename}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    print(f"\nLoaded {len(docs_data)} documents total")
    return docs_data

def extract_texts_and_metadata(docs_data: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Extract text chunks and their metadata from processed documents.
    """
    texts = []
    metadatas = []
    processed_docs = set()

    for doc in docs_data:
        document_name = doc.get('document_name', 'Unknown Document')
        if document_name in processed_docs:
            print(f"Warning: Duplicate document name found: {document_name}")
            continue
            
        processed_docs.add(document_name)
        sections = doc.get('sections', [])
        tables = doc.get('tables', [])

        # Process text sections
        for section in sections:
            content = section.get('content', '').strip()
            if not content:
                continue

            metadata = {
                'document_name': document_name,
                'document_type': section.get('metadata', {}).get('document_type', 'document'),
                'document_category': section.get('metadata', {}).get('category', 'general'),
                'section_title': section.get('title', 'No Title'),
                'section_level': str(section.get('level', 1)),
                'content_type': 'text',
                'page_number': str(section.get('metadata', {}).get('page_number', 'N/A')),
                'keywords': ','.join(section.get('keywords', [])),
                'tags': ','.join(section.get('metadata', {}).get('tags', [])),
                'language': section.get('metadata', {}).get('language', 'en'),
                'chunk_length': str(len(content))
            }

            texts.append(content)
            metadatas.append(metadata)

        # Process tables
        for table in tables:
            table_content = table.get('content', '').strip()
            if not table_content:
                continue

            metadata = {
                'document_name': document_name,
                'document_type': 'table',
                'document_category': table.get('metadata', {}).get('category', 'general'),
                'section_title': 'Table',
                'section_level': '1',
                'content_type': 'table',
                'page_number': str(table.get('metadata', {}).get('page_number', 'N/A')),
                'keywords': '',
                'tags': ','.join(table.get('metadata', {}).get('tags', [])),
                'language': table.get('metadata', {}).get('language', 'en'),
                'chunk_length': str(len(table_content))
            }

            texts.append(table_content)
            metadatas.append(metadata)

    print(f"Extracted {len(texts)} text chunks from {len(processed_docs)} documents")
    return texts, metadatas

def build_azure_search_vectorstore(texts: List[str], metadatas: List[Dict], config: Dict):
    """
    Rebuild Azure Search vector store from scratch.
    """
    print("\nInitializing Azure Search vector store...")
    
    # Initialize admin client for index operations
    admin_client = SearchIndexClient(
        endpoint=config['azure_search']['endpoint'],
        credential=AzureKeyCredential(config['azure_search']['key'])
    )
    
    # Delete existing index if it exists
    try:
        print("Deleting existing index...")
        if config['azure_search']['index_name'] in [index.name for index in admin_client.list_indexes()]:
            admin_client.delete_index(config['azure_search']['index_name'])
            print("Successfully deleted existing index")
        else:
            print("No existing index found")
    except Exception as e:
        print(f"Error during index deletion: {e}")

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Define metadata field mappings
    field_mappings = {
        'document_name': {'type': 'Edm.String', 'filterable': True, 'searchable': True},
        'document_type': {'type': 'Edm.String', 'filterable': True, 'searchable': True},
        'document_category': {'type': 'Edm.String', 'filterable': True, 'searchable': True},
        'section_title': {'type': 'Edm.String', 'filterable': True, 'searchable': True},
        'section_level': {'type': 'Edm.String', 'filterable': True, 'sortable': True},
        'content_type': {'type': 'Edm.String', 'filterable': True},
        'page_number': {'type': 'Edm.String', 'filterable': True, 'sortable': True},
        'keywords': {'type': 'Edm.String', 'filterable': True, 'searchable': True},
        'tags': {'type': 'Edm.String', 'filterable': True, 'searchable': True},
        'language': {'type': 'Edm.String', 'filterable': True},
        'chunk_length': {'type': 'Edm.String', 'filterable': True}
    }

    # Create new vector store
    vector_store = AzureSearch(
        azure_search_endpoint=config['azure_search']['endpoint'],
        azure_search_key=config['azure_search']['key'],
        index_name=config['azure_search']['index_name'],
        embedding_function=embeddings,
        metadata_field_properties=field_mappings
    )

    print(f"\nAdding {len(texts)} text chunks to fresh index...")
    vector_store.add_texts(
        texts=texts,
        metadatas=metadatas
    )
    print("Documents added successfully to new index!")
    return vector_store

def main():
    """
    Main function to orchestrate the complete index rebuild process.
    """
    PREPROCESSED_DIR = "data/preprocessed"

    try:
        # Load configuration
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(project_root)
        
        from src.config_loader import load_config, set_environment_variables
        config = load_config()
        set_environment_variables(config)

        # Load all documents
        docs_data = load_preprocessed_data(PREPROCESSED_DIR)
        
        if not docs_data:
            print("No documents found to process. Exiting...")
            return

        # Process all documents
        print("\nProcessing all documents...")
        texts, metadatas = extract_texts_and_metadata(docs_data)

        # Rebuild vector store from scratch
        build_azure_search_vectorstore(texts, metadatas, config)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
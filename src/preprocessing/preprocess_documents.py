import os
import json
import re
import unicodedata
import ftfy
from typing import List, Dict
import logging
import pandas as pd
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class DocumentPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Initialize TF-IDF Vectorizer for keyword extraction
        self.vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        # Initialize python-docx Document object
        self.doc = None

    def load_docx(self, filepath: str):
        """Load a DOCX document."""
        try:
            self.doc = Document(filepath)
        except Exception as e:
            self.logger.error(f"Failed to load document {filepath}: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean raw text content with advanced normalization."""
        # Fix text encoding issues
        text = ftfy.fix_text(text)
        text = unicodedata.normalize("NFKC", text)
        
        # Remove markup annotations like [something] or {something}
        text = re.sub(r'\[.*?\]|\{.*?\}', '', text)
        
        # Remove zero-width spaces and special chars
        text = re.sub(r'[\u00A0\u200B\u200C\u200D\u2060\uFEFF]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def extract_sections(self, filepath: str) -> List[Dict]:
        """Extract hierarchical sections with headings using python-docx."""
        self.load_docx(filepath)
        sections = []
        current_section = {'title': 'Introduction', 'level': 1, 'content': ''}
        
        for para in self.doc.paragraphs:
            if para.style.name.startswith('Heading'):
                # Extract heading level and title
                level = int(para.style.name.replace('Heading ', ''))
                title = para.text.strip()
                if not title:
                    continue  # Skip empty headings
                
                # Save the previous section
                if current_section['content']:
                    sections.append(current_section.copy())
                
                # Start a new section
                current_section = {
                    'title': title,
                    'level': level,
                    'content': ''
                }
            else:
                # Append paragraph text to current section's content
                text = para.text.strip()
                if text:
                    current_section['content'] += text + '\n'
        
        # Append the last section
        if current_section['content']:
            sections.append(current_section)
        
        return sections

    def extract_tables(self, filepath: str) -> tuple:
        """Extract tables from the DOCX file while preserving context."""
        self.load_docx(filepath)
        tables = []
        table_contexts = []
        
        for table in self.doc.tables:
            # Extract table text
            table_text = '\n'.join(['\t'.join(cell.text.strip() for cell in row.cells) for row in table.rows])
            if not table_text:
                tables.append(None)
                table_contexts.append('No context available')
                continue
            
            # Extract context surrounding the table
            # This requires knowing the table's position, which is non-trivial with python-docx
            # As a workaround, we'll associate a generic context
            context = 'Context not available due to DOCX limitations.'
            
            # Convert table text into a pipe-delimited format
            rows = table_text.split('\n')
            pipe_table = '\n'.join(['| ' + ' | '.join(row.split('\t')) + ' |' for row in rows if row])
            tables.append(pipe_table)
            table_contexts.append(context)
        
        return tables, table_contexts

    def extract_keywords(self, chunks: List[str]) -> List[str]:
        """Extract keywords for each text chunk using TF-IDF."""
        if not chunks:
            return ['No keywords']
        
        # Fit the vectorizer on the chunks
        tfidf_matrix = self.vectorizer.fit_transform(chunks)
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        keywords = []
        for row in tfidf_matrix:
            # Convert sparse row to dense array
            row_dense = row.toarray()[0]
            
            # Get indices of top 5 features with highest TF-IDF scores
            top_n_indices = row_dense.argsort()[-5:][::-1]
            
            # Retrieve the corresponding feature names
            top_features = feature_names[top_n_indices]
            
            # Extract keywords where the TF-IDF score is greater than 0
            top_keywords = [
                word for word, idx in zip(top_features, top_n_indices) if row_dense[idx] > 0
            ]
            
            # Join keywords into a comma-separated string or use a default value
            keywords.append(', '.join(top_keywords) if top_keywords else 'No keywords')
        
        return keywords

    def split_into_chunks(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks while preserving meaning."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", "  "],
            length_function=len
        )
        chunks = splitter.split_text(text)
        # Filter out empty chunks
        chunks = [c.strip() for c in chunks if c.strip()]
        return chunks

    def process_document(self, filepath: str, document_name: str) -> Dict:
        """Main processing pipeline."""
        # Extract sections
        sections = self.extract_sections(filepath)
        
        # Extract tables
        tables, table_contexts = self.extract_tables(filepath)
        
        # Split sections into chunks and extract keywords
        all_chunks = []
        all_section_titles = []
        for section in sections:
            chunks = self.split_into_chunks(section['content'])
            all_chunks.extend(chunks)
            all_section_titles.extend([section['title']] * len(chunks))
        
        # Extract keywords for all chunks
        keywords = self.extract_keywords(all_chunks)
        
        # Combine into processed_sections with metadata
        processed_sections = []
        for idx, chunk in enumerate(all_chunks):
            processed_sections.append({
                'title': all_section_titles[idx],
                'level': 1,  # Assuming top-level headings; adjust if necessary
                'content': chunk,
                'keywords': keywords[idx],
                'page_number': 'N/A'  # Placeholder; see below for handling
            })
        
        return {
            'sections': processed_sections,
            'tables': tables,
            'table_contexts': table_contexts,
            'document_name': document_name
        }

    def create_metadata(self, processed_doc: Dict) -> List[Dict]:
        """Create metadata for each chunk and table."""
        metadata = []
        
        for section in processed_doc['sections']:
            metadata_entry = {
                'keywords': section.get('keywords', 'No Keywords'),
                'section_title': section.get('title', 'No Section'),
                'section_level': section.get('level', 1),
                'content_type': 'text',
                'chunk_length': len(section.get('content', '')),
                'page_number': section.get('page_number', 'N/A'),
                'document_name': processed_doc.get('document_name', 'Unknown Document')
            }
            # Validate metadata fields
            for key in ['keywords', 'section_title', 'page_number', 'document_name']:
                if not metadata_entry[key]:
                    metadata_entry[key] = 'N/A'
            metadata.append(metadata_entry)
                
        for i, table in enumerate(processed_doc['tables']):
            metadata_entry = {
                'keywords': 'No keywords',
                'section_title': 'Table',
                'section_level': 1,
                'content_type': 'table',
                'context': processed_doc['table_contexts'][i] if processed_doc['table_contexts'] else 'No context available',
                'columns': [],  # Optional: You can extract column names if needed
                'page_number': 'N/A',  # Placeholder
                'document_name': processed_doc.get('document_name', 'Unknown Document')
            }
            # Validate metadata fields
            for key in ['keywords', 'section_title', 'page_number', 'document_name']:
                if not metadata_entry[key]:
                    metadata_entry[key] = 'N/A'
            metadata.append(metadata_entry)
                
        return metadata

def handle_element(element):
    """Convert unstructured element into a text representation suitable for DocumentPreprocessor.
    For tables, we'll attempt to convert the text into a pipe-delimited format. For images, just return [IMAGE]. 
    For normal text, just return the cleaned text.
    """
    # This function is now redundant as we're using python-docx directly
    return ""

def preprocess_all_documents(raw_data_dir="data/raw", preprocessed_dir="data/preprocessed"):
    os.makedirs(preprocessed_dir, exist_ok=True)
    preprocessor = DocumentPreprocessor()
    
    for filename in os.listdir(raw_data_dir):
        if filename.endswith(".docx"):
            filepath = os.path.join(raw_data_dir, filename)
            
            # Assume that the document name is the filename without extension
            document_name = os.path.splitext(filename)[0]
    
            processed_doc = preprocessor.process_document(filepath, document_name)
            metadata = preprocessor.create_metadata(processed_doc)
            
            # Save processed chunks and metadata as JSON
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(preprocessed_dir, f"{base_name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    'processed_doc': processed_doc,
                    'metadata': metadata
                }, f, ensure_ascii=False, indent=2)
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_all_documents()

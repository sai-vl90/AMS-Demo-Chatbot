import os
import sys
import re
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
import atexit

from transformers import T5Tokenizer
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langsmith import traceable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Local imports
from src.retrieval.pre_retrieval import PreRetrievalProcessor, PreRetrievalResult
from src.retrieval.post_retrieval import PostRetrievalProcessor, PostProcessingResult, RetrievedDocument
from src.config_loader import load_config, set_environment_variables

@dataclass
class RAGResponse:
    """Structure for final RAG response"""
    answer: str
    sources: List[Dict]
    metadata: Dict
    query_analysis: Dict
    chain_of_thought: Dict

class TokenManager:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        tokens = self.tokenizer.encode(text, max_length=max_tokens, truncation=True)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

class SafeAzureSearch(AzureSearch):
    def __del__(self):
        try:
            if sys.meta_path is not None:
                super().__del__()
        except Exception:
            pass

class RAGPipeline:
    def __init__(
        self,
        huggingface_token: str,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize the RAG pipeline with all components"""
        try:
            # Load configuration
            self.config = load_config()

            # Add token manager
            self.token_manager = TokenManager()
            # Add token limits
            self.max_input_tokens = 800  # 50 for query + 150 for context
            self.max_output_tokens = 200
            
            # Load few-shot examples
            self.few_shot_examples = self._load_few_shot_examples()
            
            # Initialize processors
            self.pre_processor = PreRetrievalProcessor()
            self.post_processor = PostRetrievalProcessor()

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

            # Initialize vector store using the SafeAzureSearch subclass
            self.vector_store = SafeAzureSearch(
                azure_search_endpoint=self.config['azure_search']['endpoint'],  # Changed to self.config
                azure_search_key=self.config['azure_search']['key'],            # Changed to self.config
                index_name=self.config['azure_search']['index_name'],           # Changed to self.config
                embedding_function=self.embeddings,
            )

            # Initialize LLM with Groq
            self.llm = ChatGroq(
                groq_api_key=self.config['groq']['api_key'],
                model_name="llama-3.3-70b-versatile",
                temperature=0.2,
                max_tokens=self.max_output_tokens,
                model_kwargs={
                    "top_p": 0.95,
                }
            )
            
            # Set up retrievers and prompts
            self._setup_retrievers()
            self._setup_prompts()

            # Register the cleanup method to be called at exit
            atexit.register(self.cleanup)  # Added atexit registration

            logger.info("RAG pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise

    def cleanup(self):
        """Explicitly clean up resources."""
        try:
            if self.vector_store:
                del self.vector_store
                self.vector_store = None
                logger.info("AzureSearch vector_store cleaned up successfully.")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    # In rag_chain.py, update the _load_few_shot_examples method:

    def _load_few_shot_examples(self) -> Dict:
        """Load few-shot examples from JSON file"""
        try:
            # Get the project root directory (2 levels up from the current file)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Construct path to configs directory at project root
            config_dir = os.path.join(project_root, "configs")
            
            # Create configs directory if it doesn't exist
            os.makedirs(config_dir, exist_ok=True)
            
            examples_path = os.path.join(config_dir, "few_shot_examples.json")
            
            # If file doesn't exist, create it with default examples
            if not os.path.exists(examples_path):
                default_examples = {
                    "examples": [
                        {
                            "type": "general",
                            "question": "What are the main features of the system?",
                            "answer": "The system includes the following key features:\n\n- **Document Processing**: Handles various document types\n- **Search Capability**: Enables efficient information retrieval\n- **Response Generation**: Provides accurate answers based on context"
                        },
                        {
                            "type": "technical",
                            "question": "How do I configure the system settings?",
                            "answer": "To configure the system:\n\n1. Access the **configuration file** in the configs directory\n2. Update the necessary parameters\n3. Save and restart the application"
                        }
                    ],
                    "formatting_rules": {
                        "headers": "Use markdown headers",
                        "lists": "Use bullet points with bold key terms",
                        "emphasis": "Use bold for important terms",
                        "tables": "Use markdown table format"
                    }
                }
                
                with open(examples_path, 'w', encoding='utf-8') as f:
                    json.dump(default_examples, f, indent=4)
                
                logger.info(f"Created default few-shot examples file at {examples_path}")
                return default_examples
                
            # Load existing file
            with open(examples_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded few-shot examples from {examples_path}")
                return data
                
        except Exception as e:
            logger.error(f"Error loading few-shot examples: {str(e)}")
            # Return empty defaults if there's an error
            return {
                "examples": [],
                "formatting_rules": {}
            }

    def _setup_retrievers(self):
        """Set up retrieval components"""
        self.base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            k=5  # Directly setting the 'k' parameter here
        )

        self.multi_retriever = MultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )

        self.compressor = LLMChainExtractor.from_llm(self.llm)

        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.multi_retriever
        )

    def _create_few_shot_prompt(self) -> str:
        """Create prompt template with few-shot examples"""
        examples = self.few_shot_examples.get("examples", [])
        formatting_rules = self.few_shot_examples.get("formatting_rules", {})
        
        # Build examples section
        examples_text = "\n\n".join([
            f"Example {i+1} - {example['type'].title()}:\n"
            f"Question: {example['question']}\n"
            f"Answer: {example['answer']}"
            for i, example in enumerate(examples[:9])  # Limit to 4 examples to manage prompt length
        ])
        
        # Build formatting rules section
        rules_text = "\n".join([
        "Follow these formatting rules strictly:",
        "1. For tables:",
        "   - Use proper markdown table syntax with aligned columns.",
        "   - Bold headers with **double asterisks**.",
        "   - Include divider rows with proper dashes.",
        "",
        "2. For lists:",
        "   - Start each item with a hyphen (-).",
        "   - Use **bold** for key terms or concepts.",
        "   - Indent sub-points with two spaces.",
        "",
        "3. For sections:",
        "   - Use **bold headers** for main sections.",
        "   - Keep consistent indentation.",
        "   - Separate sections with blank lines.",
        "",
        "4. For technical terms:",
        "   - Bold important technical terms with **double asterisks**.",
        "   - Use consistent capitalization for domain-specific terms.",
        "   - Maintain precision in numerical or coded references (e.g., GL account ranges)."
        "5. Multilingual Support:",
            "   - Always respond in the language of the question provided.",
            "   - Maintain the same formatting style for all languages.",
            "   - Translate technical terms accurately and provide English transliterations where necessary.",
    ])


        # Combine into final prompt template
        prompt_template = f"""You are an assistant that strictly provides answers based on the provided documents. If a question cannot be answered using the documents, respond with:
        "I cannot answer this question as it is outside the scope of the provided documents.". 
        Your responses must adhere to the following guidelines and formatting rules, regardless of the document's domain or context.

    Your goal:
    - Focus entirely on the provided documents and do not speculate or answer beyond their context.
    - Format responses in proper markdown style using the specified rules.
    - Adapt to the question type while maintaining professionalism and relevance.
    - Use details and examples directly from the context information.

    {rules_text}

    ### Response Rules:
    - Do not answer questions unrelated to the provided documents.
    - Use consistent formatting and style as demonstrated in the examples below.
    - Always cite the context information to support your answers.
    - Maintain a professional tone tailored to the question type.
    - Be very concise and precise in your response. Do not include unnecessary information.
    - If the user asks question about the capabilities of the model then do provide information.

    ### Example Responses:
    {examples_text}

    ### Provided Context Information:
    {{context}}

    ### Question:
    {{question}}

    ### Answer:
    """


        return prompt_template

    def _setup_prompts(self):
        """Set up prompt templates with few-shot examples"""
        prompt_template = self._create_few_shot_prompt()
        self.qa_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

    def _format_answer(self, response: str) -> str:
        """Format the response with proper structure"""
        response = response.replace('\r\n', '\n').strip()

        if ':' in response and not response.startswith('http'):
            title, content = response.split(':', 1)
            response = f"{title.strip()}:\n\n{content.strip()}"

        paragraphs = response.split('\n\n')
        formatted_paragraphs = []

        for para in paragraphs:
            lines = para.split('\n')
            formatted_lines = []

            for line in lines:
                line = line.strip()

                if line.endswith(':'):
                    formatted_lines.extend(['', line, ''])
                elif re.match(r'^\d+\.', line):
                    formatted_lines.extend(['', line])
                elif line.startswith('-'):
                    formatted_lines.extend(['', line])
                elif formatted_lines and (formatted_lines[-1].startswith('-') or
                                       re.match(r'^\d+\.', formatted_lines[-1])):
                    formatted_lines.append('    ' + line)
                else:
                    formatted_lines.append(line)

            formatted_para = '\n'.join(line for line in formatted_lines if line is not None)
            if formatted_para.strip():
                formatted_paragraphs.append(formatted_para)

        final_text = '\n\n'.join(formatted_paragraphs)
        final_text = re.sub(r'\n{3,}', '\n\n', final_text)

        return final_text.strip()

    def _format_sources(self, docs: List[RetrievedDocument]) -> List[Dict]:
        """Format source citations with metadata"""
        formatted_sources = []

        for doc in docs:
            content = doc.content.strip()
            if not content or content.lower().endswith('unknown source'):
                continue

            # Attempt to extract the source title and details
            if ' - ' in content:
                points = content.split(' - ')
                main_point = points[0]
                details = ' - '.join(points[1:]) if len(points) > 1 else ''
            else:
                main_point = content
                details = ''

            citation = {
                'content': main_point + (' - ' + details if details else '')
            }

            metadata_parts = []

            # Include the source name if available
            if doc.metadata.get('source'):
                source = doc.metadata['source']
                if source and 'unknown' not in source.lower():
                    metadata_parts.append(source)

            # Include the page number if available and valid
            page_number = doc.metadata.get('page_number')
            if page_number and isinstance(page_number, int):
                metadata_parts.append(f"Page {page_number}")
            elif page_number and isinstance(page_number, str) and page_number.isdigit():
                metadata_parts.append(f"Page {page_number}")

            # Include the section if available
            if doc.metadata.get('section'):
                metadata_parts.append(doc.metadata['section'])

            citation['metadata'] = ' | '.join(metadata_parts) if metadata_parts else 'No metadata available'
            formatted_sources.append(citation)

        return formatted_sources

    def _convert_to_retrieved_documents(self, langchain_docs: List) -> List[RetrievedDocument]:
        """Convert Langchain documents to RetrievedDocument format"""
        return [
            RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                score=doc.metadata.get('score', 0.5)
            )
            for doc in langchain_docs
        ]

    def generate_response(
        self,
        query: str,
        context: str = None
    ) -> RAGResponse:
        """Generate a response using the RAG pipeline"""
        try:
            # Truncate query if needed
            truncated_query = self.token_manager.truncate_text(query, 50)  # max 50 tokens for query

            pre_processed = self.pre_processor.process_query(truncated_query, context)
            
            logger.info(f"Query processed: {pre_processed.improved_query}")

            chain_of_thought = {
                "query_analysis": {
                    "original_query": query,
                    "improved_query": pre_processed.improved_query,
                    "reasoning": "Query improved through decomposition and enhancement"
                }
            }

            retrieved_docs = []
            sub_queries = pre_processed.sub_queries if pre_processed.sub_queries else [pre_processed.improved_query]

            for sub_query in sub_queries:
                docs = self.compression_retriever.get_relevant_documents(sub_query)
                retrieved_docs.extend(docs)

            if not retrieved_docs:
                return RAGResponse(
                    answer="I cannot answer this question as it is outside the scope of the provided documents.",
                    sources=[],
                    metadata={"reason": "No relevant documents retrieved"},
                    query_analysis={"original_query": query},
                    chain_of_thought={"reasoning": "Out-of-scope query detected"}
                )

            converted_docs = self._convert_to_retrieved_documents(retrieved_docs)
            post_processed = self.post_processor.process_documents(
                pre_processed.improved_query,
                converted_docs
            )

            chain_of_thought["document_analysis"] = {
                "num_relevant_docs": len(post_processed.reranked_documents),
                "document_types": [doc.metadata.get('source', 'unknown') for doc in post_processed.reranked_documents[:3]],
                "reasoning": "Documents ranked by relevance and technical content"
            }

            context_text = self._prepare_context(post_processed.reranked_documents)
            
            # Add token usage to metadata
            query_tokens = self.token_manager.count_tokens(truncated_query)
            context_tokens = self.token_manager.count_tokens(context_text)
            
            # Create and run the QA chain
            qa_chain = LLMChain(
                llm=self.llm,
                prompt=self.qa_prompt,
                verbose=True
                )

            qa_response = qa_chain.run({
            "context": context_text,
            "question": truncated_query
            }).strip()

            # Update metadata with token usage
            metadata = {
                "num_retrieved": len(retrieved_docs),
                "num_reranked": len(post_processed.reranked_documents),
                "processing_steps": [
                    "pre_processing",
                    "multi_query_retrieval",
                    "post_processing",
                    "reranking"
                ],
                "token_usage": {
                    "input_query_tokens": query_tokens,
                    "input_context_tokens": context_tokens,
                    "total_input_tokens": query_tokens + context_tokens,
                    "max_output_tokens": self.max_output_tokens
                }
            }

            formatted_answer = self._format_answer(qa_response)
            formatted_sources = self._format_sources(post_processed.reranked_documents[:5])

            chain_of_thought["response_generation"] = {
                "reasoning_steps": [
                    "Document type identification",
                    "Technical concept extraction",
                    "Implementation analysis",
                    "Response synthesis"
                ],
                "context_usage": "Combined relevant technical details from multiple documents"
            }

            query_analysis = {
                "original_query": query,
                "improved_query": pre_processed.improved_query,
                "sub_queries": pre_processed.sub_queries,
                "query_metadata": pre_processed.metadata
            }

            return RAGResponse(
                answer=formatted_answer,
                sources=formatted_sources,
                metadata=metadata,
                query_analysis=query_analysis,
                chain_of_thought=chain_of_thought
            )

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def _prepare_context(self, documents: List[RetrievedDocument]) -> str:
        """Prepare and truncate context to fit within token limits"""
        combined_context = "\n".join([doc.content for doc in documents[:5]])
        return self.token_manager.truncate_text(combined_context, self.max_input_tokens - 50)  # Reserve 50 tokens for query
    
# -------------------- Main Execution --------------------

if __name__ == "__main__":
    config = load_config()
    set_environment_variables(config)
    rag = RAGPipeline(
        huggingface_token=config['huggingface']['token']
    )
    try:
        query = "Who won the world cup last year?"
        response = rag.generate_response(query)
        print("\nAnswer:", response.answer)
        print("\nSources:", response.sources)
    finally:
        rag.cleanup()  # Ensures cleanup is called even if an error occurs
        rag = None  # Remove the reference to trigger garbage collection
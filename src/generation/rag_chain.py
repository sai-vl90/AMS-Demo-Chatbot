import os
import sys
import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
from functools import lru_cache

from langchain_community.vectorstores import DeepLake
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import T5Tokenizer

# Set up project path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Local imports
from src.retrieval.pre_retrieval import PreRetrievalProcessor, PreRetrievalResult
from src.retrieval.post_retrieval import PostRetrievalProcessor, PostProcessingResult, RetrievedDocument
from src.config_loader import load_config, set_environment_variables

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Structure for final RAG response"""
    answer: str
    sources: List[Dict]
    metadata: Dict
    query_analysis: Dict
    chain_of_thought: Dict

class RAGPipeline:
    def __init__(
        self,
        dataset_path: str,
        huggingface_token: str,
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """Initialize the RAG pipeline with all components"""
        try:
            # Load configuration
            self.config = load_config()
            
            # Load few-shot examples
            self.few_shot_examples = self._load_few_shot_examples()
            
            # Initialize processors
            self.pre_processor = PreRetrievalProcessor()
            self.post_processor = PostRetrievalProcessor()

            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

            # Initialize vector store
            self.vector_store = DeepLake(
                dataset_path=dataset_path,
                embedding=self.embeddings,
                read_only=True
            )

            # Initialize LLM with Groq
            self.llm = ChatGroq(
                groq_api_key=self.config['groq']['api_key'],
                model_name="gemma2-9b-it",
                temperature=0.3,
                max_tokens=8192,
                model_kwargs={
                    "top_p": 0.95,
                }
            )

            # Set up retrievers and prompts
            self._setup_retrievers()
            self._setup_prompts()

            logger.info("RAG pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {str(e)}")
            raise

    def _load_few_shot_examples(self) -> Dict:
        """Load few-shot examples from JSON file"""
        try:
            config_dir = os.path.join(project_root, "configs")
            examples_path = os.path.join(config_dir, "few_shot_examples.json")
            with open(examples_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading few-shot examples: {str(e)}")
            return {"examples": [], "formatting_rules": {}}

    def _setup_retrievers(self):
        """Set up retrieval components"""
        self.base_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
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
            for i, example in enumerate(examples[:4])  # Limit to 4 examples to manage prompt length
        ])
        
        # Build formatting rules section
        rules_text = "\n".join([
            "Follow these formatting rules strictly:",
            "1. For tables:",
            "   - Use proper markdown table syntax with aligned columns",
            "   - Bold headers with **double asterisks**",
            "   - Include divider row with proper dashes",
            "",
            "2. For lists:",
            "   - Start each item with a hyphen (-)",
            "   - Use **bold** for key terms or concepts",
            "   - Indent sub-points with two spaces",
            "",
            "3. For sections:",
            "   - Use **bold headers** for main sections",
            "   - Keep consistent indentation",
            "   - Separate sections with blank lines",
            "",
            "4. For technical terms:",
            "   - Bold important SAP and technical terms with **double asterisks**",
            "   - Use consistent capitalization for SAP modules (FI, CO, MM, etc.)",
            "   - Reference GL account ranges with proper formatting"
        ])

        # Combine into final prompt template
        prompt_template = f"""You are an expert SAP Finance analyst providing precise answers from technical documents. Your responses must use proper markdown formatting and match the style of these examples:

{examples_text}

{rules_text}

Remember:
- Keep the response focused and relevant to SAP Finance
- Use consistent formatting throughout
- Match the example style closest to the question type
- Include specific details from the context
- Maintain professional tone while adapting to question style

Context Information:
{{context}}

Question: {{question}}

Answer:"""

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
            pre_processed = self.pre_processor.process_query(query, context)
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

            context_text = "\n".join([
                doc.content for doc in post_processed.reranked_documents[:5]
            ])

            # Create and run the QA chain
            qa_chain = LLMChain(
                llm=self.llm,
                prompt=self.qa_prompt,
                verbose=True
            )
            qa_response = qa_chain.run({
                "context": context_text,
                "question": query
            }).strip()

            formatted_answer = self._format_answer(qa_response)
            formatted_sources = self._format_sources(post_processed.reranked_documents[:3])

            chain_of_thought["response_generation"] = {
                "reasoning_steps": [
                    "Document type identification",
                    "Technical concept extraction",
                    "Implementation analysis",
                    "Response synthesis"
                ],
                "context_usage": "Combined relevant technical details from multiple documents"
            }

            metadata = {
                "num_retrieved": len(retrieved_docs),
                "num_reranked": len(post_processed.reranked_documents),
                "processing_steps": [
                    "pre_processing",
                    "multi_query_retrieval",
                    "post_processing",
                    "reranking"
                ]
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

if __name__ == "__main__":
    config = load_config()
    set_environment_variables(config)
    rag = RAGPipeline(
        dataset_path=config['deeplake']['dataset_path'],
        huggingface_token=config['huggingface']['token']
    )

    query = "Who won the world cup last year?"
    response = rag.generate_response(query)
    print("\nAnswer:", response.answer)
    print("\nSources:", response.sources)
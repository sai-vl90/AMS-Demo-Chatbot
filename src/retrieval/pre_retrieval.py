import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PreRetrievalResult:
    """Structure for pre-retrieval processing results"""
    improved_query: str
    sub_queries: List[str]
    metadata: Dict

class PreRetrievalProcessor:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = None
    ):
        """
        Initialize pre-retrieval processor.
        Args:
            model_name: T5 model to use for query processing
            device: Computing device (cuda/cpu)
        """
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            
        except Exception as e:
            logger.error(f"Error initializing PreRetrievalProcessor: {str(e)}")
            raise

    def rewrite_query(self, query: str) -> str:
        """
        Improve query for better retrieval.
        Args:
            query: Original query
        Returns:
            Improved query
        """
        try:
            # Create prompt for query rewriting
            prompt = (
                "Rewrite this query to be more detailed for retrieving relevant documents. "
                "Add important context and details: " + query
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=3,
                    early_stopping=True
                )
                
            improved_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Rewrote query: {query} -> {improved_query}")
            
            return improved_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return query

    def decompose_query(self, query: str) -> List[str]:
        """
        Break complex query into simpler sub-queries.
        Args:
            query: Complex query
        Returns:
            List of simpler sub-queries
        """
        try:
            prompt = (
                "Break this complex question into simpler sub-questions. "
                "Make each sub-question self-contained: " + query
            )
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=3,
                    early_stopping=True
                )
                
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Process the result into separate sub-queries
            sub_queries = []
            for line in result.split('\n'):
                line = line.strip()
                # Remove common prefixes
                for prefix in ['- ', '• ', '1. ', '2. ', '3. ', '* ']:
                    if line.startswith(prefix):
                        line = line[len(prefix):]
                if line and not line.isspace():
                    sub_queries.append(line.strip())
                    
            logger.debug(f"Decomposed into {len(sub_queries)} sub-queries")
            return sub_queries if sub_queries else [query]
            
        except Exception as e:
            logger.error(f"Error decomposing query: {str(e)}")
            return [query]

    def is_complex_query(self, query: str) -> bool:
        """
        Determine if query needs decomposition.
        Args:
            query: Query to analyze
        Returns:
            Boolean indicating complexity
        """
        # Complexity indicators
        indicators = {
            "and",
            "compare",
            "difference",
            "versus",
            "explain",
            "describe",
            "how",
            "why",
            "what are the",
            "list all"
        }
        
        query_lower = query.lower()
        
        # Check multiple conditions
        if query_lower.count('?') > 1:
            return True
            
        if len(query_lower.split()) > 15:
            return True
            
        if any(indicator in query_lower for indicator in indicators):
            return True
            
        return False

    def process_query(
        self,
        query: str,
        context: str = None
    ) -> PreRetrievalResult:
        """
        Main method to process a query before retrieval.
        Args:
            query: Original query
            context: Optional context
        Returns:
            PreRetrievalResult with processed queries and metadata
        """
        try:
            # Track processing steps
            processing_steps = []
            
            # Add context if provided
            if context:
                query = f"Context: {context} Question: {query}"
            
            # Improve the query
            improved_query = self.rewrite_query(query)
            processing_steps.append("query_improvement")
            
            # Check complexity and decompose if needed
            sub_queries = []
            if self.is_complex_query(improved_query):
                sub_queries = self.decompose_query(improved_query)
                processing_steps.append("query_decomposition")
            
            # Prepare metadata
            metadata = {
                "original_query": query,
                "processing_steps": processing_steps,
                "num_sub_queries": len(sub_queries),
                "context_provided": bool(context)
            }
            
            return PreRetrievalResult(
                improved_query=improved_query,
                sub_queries=sub_queries,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            # Return original query if processing fails
            return PreRetrievalResult(
                improved_query=query,
                sub_queries=[],
                metadata={"error": str(e)}
            )

if __name__ == "__main__":
    # Test the processor
    processor = PreRetrievalProcessor()
    
    # Test queries
    test_queries = [
        "What are the controlling features in SAP?",
        "Compare and explain the differences between cost center accounting and profit center accounting in the finance blueprint.",
        "How does SAP handle internal orders and what are their main uses?"
    ]
    
    for query in test_queries:
        print(f"\nOriginal Query: {query}")
        result = processor.process_query(query)
        print(f"Improved Query: {result.improved_query}")
        if result.sub_queries:
            print("Sub-queries:")
            for i, sq in enumerate(result.sub_queries, 1):
                print(f"{i}. {sq}")
        print("Metadata:", result.metadata)
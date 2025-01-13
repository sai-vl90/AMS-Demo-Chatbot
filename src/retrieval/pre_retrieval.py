import torch
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
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
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = None):
        """
        Initialize pre-retrieval processor.
        Args:
            model_name: T5 model to use for query processing
            device: Computing device (cuda/cpu)
        """
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            logger.error(f"Error initializing PreRetrievalProcessor: {str(e)}")
            raise

    def is_complex_query(self, query: str) -> bool:
        """
        Determine if a query needs decomposition based on complexity indicators.
        Args:
            query: Query to analyze
        Returns:
            Boolean indicating complexity
        """
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
        if query_lower.count('?') > 1:
            return True
        if len(query_lower.split()) > 15:
            return True
        if any(indicator in query_lower for indicator in indicators):
            return True
        return False

    def rewrite_query(self, query: str, domain_context: str = None) -> str:
        """
        Improve query for better retrieval with expanded domain-specific prompts.
        Args:
            query: Original query
            domain_context: Context of the query's domain
        Returns:
            Improved query
        """
        try:
            domain_prompts = {
                "general": "Be detailed, precise, and include relevant context for improved information retrieval."
            }
            context_prompt = domain_prompts.get(domain_context, domain_prompts["general"])
            prompt = f"You are an expert query optimizer. Rewrite the query to be detailed, precise, and aligned with: {context_prompt}\nOriginal Query: {query}"
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(inputs["input_ids"], max_length=100, num_beams=5, early_stopping=True)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return query

    def extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from the query using NER.
        Args:
            query: Original query
        Returns:
            List of entities
        """
        try:
            entities = self.ner_pipeline(query)
            extracted_entities = [
                entity['word'] for entity in entities if entity.get('entity') in {"ORG", "LOC", "PERSON", "MISC"}
            ]
            logger.info(f"Extracted entities: {extracted_entities}")
            return extracted_entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

    def decompose_query(self, query: str, domain_context: str = None) -> List[str]:
        """
        Break a complex query into simpler sub-queries with entity extraction and logical parsing.
        Args:
            query: Complex query
            domain_context: Optional domain context for tailored decomposition
        Returns:
            List of simpler sub-queries
        """
        try:
            entities = self.extract_entities(query)
            if not entities:
                logger.warning("No entities extracted. Decomposition will rely solely on logical clauses.")

            logical_clauses = [clause.strip() for clause in re.split(r'\band\b|\bor\b|\bversus\b', query, flags=re.IGNORECASE) if clause.strip()]
            logger.info(f"Identified logical clauses: {logical_clauses}")

            sub_queries = []
            for clause in logical_clauses:
                if entities:
                    sub_queries.append(f"{clause}: Focus on {'; '.join(entities)}")
                else:
                    sub_queries.append(clause)

            return sub_queries if sub_queries else [query]
        except Exception as e:
            logger.error(f"Error decomposing query: {str(e)}")
            return [query]

    def validate_sub_queries(self, sub_queries: List[str]) -> List[str]:
        """
        Validate sub-queries to ensure they are standalone and meaningful.
        Args:
            sub_queries: List of generated sub-queries
        Returns:
            Refined list of sub-queries
        """
        refined_sub_queries = []
        for sq in sub_queries:
            if len(sq.split()) > 3:  # Ensure sufficient length
                refined_sub_queries.append(sq)
        return refined_sub_queries

    def process_query(self, query: str, context: str = None) -> PreRetrievalResult:
        """
        Main method to process a query before retrieval.
        Args:
            query: Original query
            context: Optional context
        Returns:
            PreRetrievalResult with processed queries and metadata
        """
        try:
            processing_steps = []

            if context:
                query = f"Context: {context} Question: {query}"

            improved_query = self.rewrite_query(query, context)
            processing_steps.append("query_improvement")

            sub_queries = []
            if self.is_complex_query(improved_query):
                sub_queries = self.decompose_query(improved_query, context)
                sub_queries = self.validate_sub_queries(sub_queries)
                processing_steps.append("query_decomposition")

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
            return PreRetrievalResult(
                improved_query=query,
                sub_queries=[],
                metadata={"error": str(e)}
            )

if __name__ == "__main__":
    processor = PreRetrievalProcessor()

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

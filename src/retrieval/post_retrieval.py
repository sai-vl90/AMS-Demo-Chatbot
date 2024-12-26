import torch
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievedDocument:
    """Structure for a retrieved document"""
    content: str
    metadata: Dict
    score: float = 0.0

@dataclass
class PostProcessingResult:
    """Structure for post-processing results"""
    reranked_documents: List[RetrievedDocument]
    summary: str
    metadata: Dict

class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = None
    ):
        """Initialize the BGE reranker"""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def compute_scores(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """
        Compute relevance scores for documents using batched processing
        """
        scores = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Prepare inputs
            pairs = [[query, doc] for doc in batch_docs]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Calculate scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.softmax(dim=-1)[:, 1].cpu().numpy()
                scores.extend(batch_scores.tolist())
        
        return scores

class PostRetrievalProcessor:
    def __init__(
        self,
        reranker_model: str = "BAAI/bge-reranker-base",
        summary_model: str = "google/flan-t5-base",
        device: str = None
    ):
        """
        Initialize post-retrieval processor with BGE reranker
        Args:
            reranker_model: BGE reranker model name
            summary_model: Model for summarization
            device: Computing device (cuda/cpu)
        """
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Initialize BGE reranker
            self.reranker = BGEReranker(reranker_model, self.device)
            
            # Initialize summarization model (keeping T5 for this)
            self.tokenizer = AutoTokenizer.from_pretrained(summary_model)
            self.summary_model = AutoModelForSequenceClassification.from_pretrained(
                summary_model
            ).to(self.device)
            
        except Exception as e:
            logger.error(f"Error initializing PostRetrievalProcessor: {str(e)}")
            raise

    def rerank_documents(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using BGE reranker
        Args:
            query: Original query
            documents: Retrieved documents
            top_k: Number of documents to return
        Returns:
            Reranked list of documents
        """
        try:
            if not documents:
                return []
            
            # Extract document contents
            doc_contents = [doc.content for doc in documents]
            
            # Get relevance scores using BGE reranker
            scores = self.reranker.compute_scores(query, doc_contents)
            
            # Update document scores
            for doc, score in zip(documents, scores):
                doc.score = float(score)
            
            # Sort by score and return top_k
            reranked = sorted(documents, key=lambda x: x.score, reverse=True)
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Error in document reranking: {str(e)}")
            return documents[:top_k]

    def process_documents(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> PostProcessingResult:
        """
        Main method to process retrieved documents
        Args:
            query: Original query
            documents: Retrieved documents
        Returns:
            PostProcessingResult with processed documents and metadata
        """
        try:
            processing_steps = []
            
            # Rerank documents using BGE reranker
            reranked_docs = self.rerank_documents(query, documents)
            processing_steps.append("bge_reranking")
            
            # Generate summary
            summary = self.generate_summary(query, reranked_docs)
            processing_steps.append("summarization")
            
            # Prepare metadata
            metadata = {
                "initial_docs": len(documents),
                "reranked_docs": len(reranked_docs),
                "processing_steps": processing_steps,
                "avg_score": sum(doc.score for doc in reranked_docs) / len(reranked_docs) if reranked_docs else 0,
                "reranker_model": "BAAI/bge-reranker-base"
            }
            
            return PostProcessingResult(
                reranked_documents=reranked_docs,
                summary=summary,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            return PostProcessingResult(
                reranked_documents=documents[:5],
                summary="Error processing documents.",
                metadata={"error": str(e)}
            )

    def generate_summary(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> str:
        """Generate a query-focused summary of documents"""
        # Implementation remains similar to your original version
        try:
            # Combine top documents
            combined_text = " ".join([
                doc.content for doc in documents[:3]
            ])
            
            # Prepare summarization prompt
            prompt = (
                f"Summarize the following information to answer the query: {query}\n\n"
                f"Information: {combined_text}"
            )
            
            inputs = self.tokenizer(
                prompt,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.summary_model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
                
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary."
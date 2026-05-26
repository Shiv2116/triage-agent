"""
retriever.py - BM25-based document retrieval system.

Indexes corpus documents and retrieves relevant ones based on queries.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from rank_bm25 import BM25Okapi

from config import (
    DATA_DIR,
    DEVPLATFORM_CORPUS,
    CLAUDE_CORPUS,
    VISA_CORPUS,
    BM25_TOP_K,
)
from utils import tokenize

logger = logging.getLogger(__name__)


class Document:
    """Represents a single document."""

    def __init__(self, path: str, content: str, product: str = "unknown"):
        self.path = path
        self.content = content
        self.product = product
        self.tokens = content.lower().split()


class BM25Retriever:
    """BM25-based retriever for support documents."""

    def __init__(self):
        """Initialize retriever."""
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus_loaded = False

    def load_corpus(self) -> int:
        """
        Load all corpus documents.

        Returns:
            Number of documents loaded
        """
        if self.corpus_loaded:
            logger.info("Corpus already loaded")
            return len(self.documents)

        logger.info("Loading corpus...")
        doc_count = 0

        # Load from each product directory
        for corpus_dir, product in [
            (DEVPLATFORM_CORPUS, "DevPlatform"),
            (CLAUDE_CORPUS, "Claude"),
            (VISA_CORPUS, "Visa"),
        ]:
            doc_count += self._load_corpus_directory(corpus_dir, product)

        # Build BM25 index
        self._build_index()
        self.corpus_loaded = True
        logger.info(f"Corpus loaded: {doc_count} documents")

        return doc_count

    def _load_corpus_directory(self, corpus_dir: Path, product: str) -> int:
        """Load all markdown files from a directory recursively."""
        count = 0

        if not corpus_dir.exists():
            logger.warning(f"Corpus directory not found: {corpus_dir}")
            return 0

        for md_file in corpus_dir.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if content.strip():
                    # Calculate relative path for output
                    rel_path = str(md_file.relative_to(DATA_DIR.parent))

                    doc = Document(
                        path=rel_path,
                        content=content,
                        product=product,
                    )
                    # Use centralized tokenize utility
                    doc.tokens = tokenize(content)
                    self.documents.append(doc)
                    count += 1

            except Exception as e:
                logger.error(f"Error loading {md_file}: {e}")

        logger.info(f"Loaded {count} documents from {product}")
        return count

    def _build_index(self) -> None:
        """Build BM25 index from loaded documents."""
        if not self.documents:
            logger.warning("No documents to index")
            return

        # Prepare tokenized corpus
        corpus = [doc.tokens for doc in self.documents]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"Built BM25 index for {len(corpus)} documents")

    def retrieve(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        product_filter: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve top-k relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return
            product_filter: Optional product filter

        Returns:
            List of (doc_path, score, content) tuples
        """
        if not self.bm25:
            logger.error("BM25 index not built")
            return []

        if not query or len(query.strip()) == 0:
            logger.warning("Empty query")
            return []

        try:
            # Tokenize query
            query_tokens = query.lower().split()

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get top-k indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:top_k]

            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]

                    # Apply product filter if specified
                    if product_filter and doc.product != product_filter:
                        continue

                    score = scores[idx]
                    results.append((doc.path, score, doc.content))

            logger.info(f"Retrieved {len(results)} documents for query")
            return results

        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []

    def retrieve_by_product(self, query: str, product: str) -> List[Tuple[str, float, str]]:
        """
        Retrieve documents filtered by product.

        Args:
            query: Search query
            product: Product name filter

        Returns:
            List of (doc_path, score, content) tuples
        """
        return self.retrieve(query, product_filter=product)

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search and return structured results.

        Returns:
            List of dicts with path, score, and snippet
        """
        results = self.retrieve(query, top_k=top_k)

        structured = []
        for path, score, content in results:
            # Generate snippet
            snippet = self._generate_snippet(content, query, max_chars=200)

            structured.append({
                "path": path,
                "score": round(float(score), 4),
                "snippet": snippet,
                "full_content": content,
            })

        return structured

    def _generate_snippet(self, content: str, query: str, max_chars: int = 200) -> str:
        """Generate a snippet of content around query terms."""
        content_lower = content.lower()
        query_lower = query.lower()

        # Find first occurrence of any query term
        for term in query_lower.split():
            idx = content_lower.find(term)
            if idx != -1:
                start = max(0, idx - 50)
                end = min(len(content), start + max_chars)
                snippet = content[start:end].replace("\n", " ")
                return f"...{snippet}..."

        # Fallback: first max_chars
        return content[:max_chars] + "..."

    def get_document_count(self) -> int:
        """Get total number of documents loaded."""
        return len(self.documents)

    def get_products(self) -> List[str]:
        """Get unique products in corpus."""
        return sorted(list(set(doc.product for doc in self.documents)))

    def validate_paths(self, paths: List[str]) -> List[str]:
        """
        Validate that given paths exist in corpus.

        Returns:
            List of valid paths
        """
        valid_paths = []
        loaded_paths = {doc.path for doc in self.documents}

        for path in paths:
            if path in loaded_paths:
                valid_paths.append(path)
            else:
                logger.warning(f"Invalid corpus path: {path}")

        return valid_paths


def create_retriever() -> BM25Retriever:
    """Factory function to create and load retriever."""
    retriever = BM25Retriever()
    retriever.load_corpus()
    return retriever

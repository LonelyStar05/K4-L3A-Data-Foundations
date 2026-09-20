"""Advanced retrieval pipeline built on top of the lab's `src` package.

corpus     -> load data/<topic>/*.md (frontmatter -> metadata, body -> content)
chunkers   -> StructureAwareChunker (heading tree + contextual headers) and the lab baselines
embeddings -> CachedOpenAIEmbedder (batched, disk-cached, plugs into src.EmbeddingStore)
bm25       -> sparse retrieval with Vietnamese syllable bigrams
rerank     -> cross-encoder reranker (BAAI/bge-reranker-v2-m3 by default)
llm        -> cached OpenAI chat wrapper (HyDE + answer generation)
retriever  -> HybridRetriever: dense + HyDE + BM25 -> RRF -> rerank, metadata pre-filter
pipeline   -> RAGPipeline: retrieve -> cited answer
"""

from .chunkers import BaselineChunker, StructureAwareChunker, chunk_corpus
from .corpus import SourceDoc, load_corpus
from .pipeline import RAGPipeline
from .retriever import HybridRetriever

__all__ = [
    "SourceDoc",
    "load_corpus",
    "StructureAwareChunker",
    "BaselineChunker",
    "chunk_corpus",
    "HybridRetriever",
    "RAGPipeline",
]

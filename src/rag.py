import os
import uuid
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from src.classifier import get_classifier

load_dotenv()


class RAGManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = os.path.abspath(persist_directory)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self.embedding_fn
        )

    def _get_splitter(self, doc_name: str):
        ext = os.path.splitext(doc_name)[1].lower()

        language_map = {
            ".py": Language.PYTHON,
            ".js": Language.JS,
            ".jsx": Language.JS,
            ".ts": Language.TS,
            ".tsx": Language.TS,
            ".c": Language.CPP,
            ".cpp": Language.CPP,
            ".cc": Language.CPP,
            ".cxx": Language.CPP,
            ".h": Language.CPP,
            ".hpp": Language.CPP,
            ".java": Language.JAVA,
            ".cs": Language.CSHARP,
            ".csx": Language.CSHARP,
            ".html": Language.HTML,
            ".htm": Language.HTML,
        }

        if ext in language_map:
            return RecursiveCharacterTextSplitter.from_language(
                language=language_map[ext],
                chunk_size=1000,
                chunk_overlap=150
            )

        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

    def add_document(self, text: str, source: str = "user-feedback", extra_meta: dict = None) -> int:
        splitter = self._get_splitter(source)
        chunks = splitter.split_text(text)
        if not chunks:
            return 0

        unique_prefix = uuid.uuid4().hex[:8]
        ids = [f"{source}_{unique_prefix}_{i}" for i in range(len(chunks))]

        base_meta = {"source": source}
        if extra_meta:
            for k, v in extra_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    base_meta[k] = v
                else:
                    base_meta[k] = str(v)

        metadatas = [base_meta.copy() for _ in chunks]

        self.collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        total = self.count()
        if total == 0:
            return []
        n_results = min(k, total)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        docs = []
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        for doc, meta in zip(documents, metadatas):
            docs.append({
                "text": doc,
                "meta": meta or {"source": "unknown"}
            })
        return docs

    def is_relevant(self, query: str, docs: list[dict]) -> bool:
        if not docs:
            return False
        context = "\n\n---\n\n".join(
            f"[source: {d['meta'].get('source', 'unknown')}]\n{d['text']}" for d in docs
        )
        try:
            classifier = get_classifier()
            return classifier.check_relevance(query, context)
        except Exception:
            query_words = set(query.lower().split())
            context_words = set(context.lower().split())
            overlap = query_words.intersection(context_words)
            return len(overlap) > 0


_rag_manager_instance = None


def get_rag_manager() -> RAGManager:
    global _rag_manager_instance
    if _rag_manager_instance is None:
        _rag_manager_instance = RAGManager()
    return _rag_manager_instance


def retrieve(query: str, k: int = 4) -> list[dict]:
    return get_rag_manager().retrieve(query, k=k)


def is_relevant(query: str, docs: list[dict]) -> bool:
    return get_rag_manager().is_relevant(query, docs)


def count() -> int:
    return get_rag_manager().count()


def add_document(text: str, source: str = "user-feedback", extra_meta: dict = None) -> int:
    return get_rag_manager().add_document(text, source=source, extra_meta=extra_meta)
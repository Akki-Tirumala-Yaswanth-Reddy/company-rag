import os

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    HeadingHierarchyOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()


class MarkdownTables(ChunkingSerializerProvider):
    """Tell the chunker to write tables as Markdown instead of 'row, column = value' sentences."""
    def get_serializer(self, doc):
        return ChunkingDocSerializer(doc=doc, table_serializer=MarkdownTableSerializer())
    # Default chunker for tables was clipping the names of columns


chunker = HybridChunker(serializer_provider=MarkdownTables())

DOCS_DIR = os.getenv("DOCS_DIR","data/docs")
MILVUS_URI = os.getenv("MILVUS_DB_PATH", "./milvus.db")
COLLECTION = os.getenv("COLLECTION", "acme_v2")
USE_MODEL = os.getenv("USE_MODEL", "GOOGLE")
USE_EMBEDDING_MODEL = os.getenv("USE_EMBEDDING_MODEL", "GOOGLE")
JUDGE_LLM = os.getenv("JUDGE_LLM", "GOOGLE")

pipeline_options = PdfPipelineOptions()
pipeline_options.heading_hierarchy_options = HeadingHierarchyOptions(enabled=True)
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
# I used qwen as the application was hitting google's RPM limits
if USE_EMBEDDING_MODEL == "GOOGLE":
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
else:
    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:4b",
        keep_alive=5,
    )
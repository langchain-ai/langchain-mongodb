"""langchain_mongodb_deepagents_vfs: MongoDB Atlas-backed filesystem adapter for LangChain DeepAgents."""

from langchain_mongodb_deepagents_vfs.backend import MongoFilesystemBackend
from langchain_mongodb_deepagents_vfs.errors import AdapterError, ErrorCode

__version__ = "0.1.0"
__all__ = ["MongoFilesystemBackend", "AdapterError", "ErrorCode"]

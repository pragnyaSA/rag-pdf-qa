"""Runs before app modules are imported: keep tests offline and independent of a local .env."""
import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")

"""
API package initializer for chatbot_backend.src.api.

Exports convenience accessors for background processing.
"""

# Re-export background job manager accessor for consumers who import from src.api directly.
from .background_jobs import get_job_manager  # noqa: F401

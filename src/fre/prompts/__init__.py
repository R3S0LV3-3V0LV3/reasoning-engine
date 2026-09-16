"""Versioned prompt definitions."""

from fre.prompts.registry import PromptRegistry, default_prompt_registry
from fre.prompts.schemas import OutputSchemaRegistry, default_output_schema_registry

__all__ = [
    "OutputSchemaRegistry",
    "PromptRegistry",
    "default_output_schema_registry",
    "default_prompt_registry",
]

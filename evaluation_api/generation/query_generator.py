# --- File: search-evaluation-api/generation/query_generator.py ---
# This is a STUBBED module. It simulates LLM calls to show the pipeline structure.

import logging
import os
from typing import List
from tqdm import tqdm

from .models import SelectionBundle, GeneratedQuery

logger = logging.getLogger(__name__)

class QueryGenerator:
    def __init__(self, config):
        """
        Initialize Azure OpenAI client. No fallback.
        """
        self.config = config
        self.client = None
        try:
            from openai import AzureOpenAI  # type: ignore
            endpoint = getattr(self.config, "AZURE_OPENAI_ENDPOINT", None)
            api_key = (
                os.environ.get("AZURE_OPENAI_KEY")
                or os.environ.get("AZURE_OPENAI_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            if not endpoint:
                logger.error("AZURE_OPENAI_ENDPOINT is not set. Please set it in .env.")
                raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT")
            if not api_key:
                logger.error("Azure OpenAI key not set. Please set AZURE_OPENAI_KEY (or AZURE_OPENAI_API_KEY/OPENAI_API_KEY) in .env.")
                raise RuntimeError("Missing Azure OpenAI API key")

            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-02-01"
            )
            logger.info("QueryGenerator initialized with Azure OpenAI endpoint.")
        except ImportError:
            logger.error("openai package not installed. Please add 'openai' to requirements and install.")
            raise

    def generate_queries(self, bundles: List[SelectionBundle]) -> List[GeneratedQuery]:
        """Generates multiple query types for each bundle."""
        if self.client is None:
            raise RuntimeError("Azure OpenAI client not initialized")

        all_queries = []
        for bundle in tqdm(bundles, desc="Generating queries"):
            for query_type in self.config.QUERY_TYPES:
                # This check prevents trying to make a "comparison" query
                # from a single golden chunk.
                if query_type == "comparison" and len(bundle.golden_chunks) < 2:
                    continue
                    
                prompt = self._build_prompt(bundle, query_type)

                # Real LLM only; raise on failure
                query_text = self._call_llm(prompt)
                
                if query_text:
                    all_queries.append(
                        GeneratedQuery(
                            query=query_text,
                            golden_chunks=bundle.golden_chunks,
                            query_type=query_type,
                            distractor_chunks=bundle.distractor_chunks
                        )
                    )
        logger.info("Generated %s queries.", len(all_queries))
        return all_queries

    def _build_prompt(self, bundle: SelectionBundle, query_type: str) -> str:
        """Constructs the distractor-aware prompt."""
        
        golden_context = "\n---\n".join([c.chunk_text for c in bundle.golden_chunks])
        distractor_context = "\n---\n".join([c.chunk_text for c in bundle.distractor_chunks])

        prompt = f"""
        You are a search query generation expert. Your task is to generate a high-quality, {query_type} search query.
        
        The query MUST be answerable ONLY by the 'Golden Context'.
        The query MUST NOT be answerable by the 'Distractor Context'.
        
        ## Golden Context:
        {golden_context}
        
        ## Distractor Context:
        {distractor_context}
        
        ## Task:
        Generate a single, concise {query_type} query based *only* on the Golden Context.
        
        Query:
        """
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Real Azure OpenAI chat.completions call."""
        if self.client is None:
            raise RuntimeError("Azure OpenAI client not initialized")
        model = getattr(self.config, "AZURE_OPENAI_DEPLOYMENT_NAME", None)
        if not model:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT_NAME not set in config")
        temperature = getattr(self.config, "TEMPERATURE", 0.7)
        max_tokens = getattr(self.config, "MAX_TOKENS", 64)

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

# --- End File: search-evaluation-api/generation/query_generator.py ---

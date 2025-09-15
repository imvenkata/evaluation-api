# --- File: search-evaluation-api/generation/query_generator.py ---
# This is a STUBBED module. It simulates LLM calls to show the pipeline structure.

import logging
from typing import List
from tqdm import tqdm

from .models import SelectionBundle, GeneratedQuery

logger = logging.getLogger(__name__)

class QueryGenerator:
    def __init__(self, config):
        """
        In a real implementation, this is where you would initialize
        your Azure OpenAI client using config.AZURE_OPENAI_ENDPOINT etc.
        """
        self.config = config
        logger.info("QueryGenerator initialized (STUBBED).")

    def generate_queries(self, bundles: List[SelectionBundle]) -> List[GeneratedQuery]:
        """Generates multiple query types for each bundle."""
        all_queries = []
        for bundle in tqdm(bundles, desc="Generating queries"):
            for query_type in self.config.QUERY_TYPES:
                # This check prevents trying to make a "comparison" query
                # from a single golden chunk.
                if query_type == "comparison" and len(bundle.golden_chunks) < 2:
                    continue
                    
                prompt = self._build_prompt(bundle, query_type)
                
                # --- STUBBED LLM CALL ---
                query_text = self._call_stubbed_llm(bundle, query_type)
                # --- END STUB ---
                
                if query_text:
                    all_queries.append(
                        GeneratedQuery(
                            query=query_text,
                            golden_chunks=bundle.golden_chunks,
                            query_type=query_type
                        )
                    )
        logger.info(f"Generated {len(all_queries)} queries (STUBBED).")
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

    def _call_stubbed_llm(self, bundle: SelectionBundle, query_type: str) -> str:
        """A stub that simulates an LLM call."""
        first_chunk_text = bundle.golden_chunks[0].chunk_text[:50]
        return f"MOCK {query_type.upper()} QUERY about '{first_chunk_text}...'"

    def _call_llm(self, prompt: str) -> str:
        """This is where the real Azure OpenAI call would go."""
        # import os
        # from openai import AzureOpenAI
        # client = AzureOpenAI(
        #     azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
        #     api_key=os.environ["AZURE_OPENAI_KEY"],
        #     api_version="2024-02-01"
        # )
        # response = client.chat.completions.create(
        #     model=self.config.AZURE_OPENAI_DEPLOYMENT_NAME,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=30,
        #     temperature=0.7
        # )
        # return response.choices[0].message.content
        raise NotImplementedError("LLM call is stubbed. Use _call_stubbed_llm for testing.")

# --- End File: search-evaluation-api/generation/query_generator.py ---

# --- File: search-evaluation-api/generation/query_generator.py ---
# This is a STUBBED module. It simulates LLM calls to show the pipeline structure.

import logging
import os
import random
import re
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from tqdm import tqdm

from .models import SelectionBundle, GeneratedQuery
from ..utils.cache_utils import SimpleCache, create_prompt_cache_key, create_config_hash

logger = logging.getLogger(__name__)

class QueryGenerator:
    def __init__(self, config):
        """
        Initialize Azure OpenAI client and caching system.
        """
        self.config = config
        self.client = None
        
        # Initialize caching if enabled
        self.cache_enabled = getattr(config, 'CACHE_LLM_QUERIES', True) and getattr(config, 'ENABLE_CACHING', True)
        if self.cache_enabled:
            cache_dir = getattr(config, 'CACHE_DIR', './cache')
            self.cache = SimpleCache(cache_dir, namespace="llm_queries")
            self.config_hash = create_config_hash(config)
            logger.info("LLM query caching enabled: %s", self.cache.cache_dir)
        else:
            self.cache = None
            logger.info("LLM query caching disabled")
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

        max_workers = int(getattr(self.config, "LLM_MAX_WORKERS", 16))
        burst_capacity = int(getattr(self.config, "LLM_BURST_CAPACITY", 50))
        refill_rate = float(getattr(self.config, "LLM_REFILL_RATE", 10.0))

        # Enhanced token bucket with burst capacity
        rate_limit_lock = threading.Lock()
        token_bucket = {"tokens": burst_capacity, "last_refill": time.time()}

        def acquire_token():
            nonlocal token_bucket
            while True:
                with rate_limit_lock:
                    now = time.time()
                    # Refill tokens based on time elapsed
                    elapsed = now - token_bucket["last_refill"]
                    token_bucket["tokens"] = min(
                        burst_capacity, 
                        token_bucket["tokens"] + (elapsed * refill_rate)
                    )
                    token_bucket["last_refill"] = now
                    
                    if token_bucket["tokens"] >= 1.0:
                        token_bucket["tokens"] -= 1.0
                        return
                
                # Adaptive sleep based on token deficit
                sleep_time = max(0.01, (1.0 - token_bucket["tokens"]) / refill_rate)
                time.sleep(min(sleep_time, 0.1))

        def process(bundle: SelectionBundle, raw_query_type: str):
            query_type = self._normalize_query_type(raw_query_type)
            if query_type == "comparison" and len(bundle.golden_chunks) < 2:
                return None
            
            # Try cache first if enabled
            query_text = None
            if self.cache_enabled and self.cache:
                query_text = self._get_cached_query(bundle, query_type)
            
            # If not cached, generate new query
            if query_text is None:
                prompt, max_tokens_override = self._build_prompt(bundle, query_type)
                # Rate limit and call LLM with retry
                from tenacity import retry, stop_after_attempt, wait_exponential_jitter  # type: ignore

                @retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=0.2, max=6.0))
                def call():
                    acquire_token()
                    return self._call_llm(prompt, max_tokens_override=max_tokens_override)

                try:
                    query_text = call()
                    # Cache the result if enabled
                    if self.cache_enabled and self.cache and query_text:
                        self._cache_query(bundle, query_type, query_text)
                except (RuntimeError, ValueError, TimeoutError) as e:
                    # Handle specific Azure OpenAI errors
                    logger.warning("LLM call failed after retries: %s", e)
                    return None
            
            if query_text:
                cleaned = self._postprocess_query(query_text, query_type, bundle)
                return GeneratedQuery(
                    query=cleaned,
                    golden_chunks=bundle.golden_chunks,
                    query_type=query_type,
                    distractor_chunks=bundle.distractor_chunks,
                )
            return None

        tasks = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for bundle in bundles:
                for qt in self.config.QUERY_TYPES:
                    tasks.append(ex.submit(process, bundle, qt))
            results = []
            for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Generating queries"):
                res = fut.result()
                if res is not None:
                    results.append(res)
        logger.info("Generated %s queries.", len(results))
        return results

    def _build_prompt(self, bundle: SelectionBundle, query_type: str) -> Tuple[str, int]:
        """Construct a type-specific, distractor-aware prompt and return (prompt, max_tokens)."""

        golden_context = "\n---\n".join([c.chunk_text for c in bundle.golden_chunks])
        distractor_context = "\n---\n".join([c.chunk_text for c in bundle.distractor_chunks])

        # Per-type constraints and guidance
        constraints: List[str] = [
            "Answerable ONLY by the Golden Context.",
            "NOT answerable by the Distractor Context.",
        ]

        max_tokens_map = getattr(self.config, "QUERY_TYPE_MAX_TOKENS", {}) or {}
        default_max = int(getattr(self.config, "MAX_TOKENS", 32))
        max_tokens_for_type = int(max_tokens_map.get(query_type, default_max))

        length_targets = getattr(self.config, "QUERY_LENGTH_TARGETS", {}) or {}
        long_range = length_targets.get("long", (21, 128))
        med_range = length_targets.get("medium", (5, 20))
        short_range = length_targets.get("short", (1, 4))

        guidance = ""
        if query_type == "concept_seeking":
            guidance = (
                "Ask an abstract 'why'/'how' style question that typically requires multiple sentences to answer. "
                "Avoid copying long phrases; focus on higher-level concepts."
            )
        elif query_type == "exact_snippet":
            guidance = (
                "Produce a longer query that is an EXACT contiguous substring from the Golden Context. "
                "Copy the text verbatim from the Golden Context. Do NOT paraphrase."
            )
            constraints.append("Must be a verbatim substring present in Golden Context.")
        elif query_type == "web_search_like":
            guidance = (
                "Produce a short, search-engine-style query (2-6 words) that a user might type."
            )
        elif query_type == "low_overlap":
            guidance = (
                "Paraphrase using different words than the Golden Context while preserving meaning. "
                "Target low lexical overlap (<30%) with the Golden Context."
            )
        elif query_type == "fact_seeking":
            guidance = (
                "Ask a single, clear fact-seeking question (e.g., who/what/when/how many)."
            )
        elif query_type == "keyword":
            guidance = (
                "Output only the key identifier words (nouns/proper nouns), no stopwords. 1-4 tokens."
            )
        elif query_type == "misspellings":
            guidance = (
                "Write the query with 1-2 realistic typos/transpositions/leet substitutions while keeping intent clear."
            )
        elif query_type == "long":
            guidance = (
                f"Write a verbose query longer than {long_range[0]} tokens."
            )
        elif query_type == "medium":
            guidance = (
                f"Write a medium-length query between {med_range[0]} and {med_range[1]} tokens."
            )
        elif query_type == "short":
            guidance = (
                f"Write a concise query with {short_range[0]}-{short_range[1]} tokens."
            )
        elif query_type == "comparison":
            guidance = (
                "Ask a question that requires comparing or contrasting at least two Golden Context snippets."
            )
        else:
            guidance = f"Generate a high-quality {query_type} search query."

        constraints_text = "\n- ".join(["Constraints:"] + constraints)

        prompt = (
            f"You are a search query generation expert.\n\n"
            f"Task: Generate ONE {query_type} query.\n"
            f"{constraints_text}\n\n"
            f"Guidance: {guidance}\n\n"
            f"Golden Context:\n{golden_context}\n\n"
            f"Distractor Context:\n{distractor_context}\n\n"
            f"Output ONLY the query, no quotes, no explanations."
        )
        return prompt, max_tokens_for_type

    def _call_llm(self, prompt: str, max_tokens_override: int = None) -> str:
        """Optimized Azure OpenAI chat.completions call with timeout and error handling."""
        if self.client is None:
            raise RuntimeError("Azure OpenAI client not initialized")
        model = getattr(self.config, "AZURE_OPENAI_DEPLOYMENT_NAME", None)
        if not model:
            raise RuntimeError("AZURE_OPENAI_DEPLOYMENT_NAME not set in config")
        
        # Optimized parameters for faster generation
        temperature = getattr(self.config, "TEMPERATURE", 0.5)
        max_tokens = int(max_tokens_override or getattr(self.config, "MAX_TOKENS", 32))
        timeout_seconds = getattr(self.config, "LLM_TIMEOUT_SECONDS", 15)

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_seconds,
                # Additional optimizations
                top_p=0.9,  # Slightly reduce randomness for faster generation
                frequency_penalty=0.1,  # Encourage variety
                presence_penalty=0.1
            )
            return (response.choices[0].message.content or "").strip()
        except (RuntimeError, ValueError, TimeoutError) as e:
            # Log but don't fail the entire batch
            logger.warning("LLM call failed: %s", str(e)[:100])
            raise

    def _normalize_query_type(self, qt: str) -> str:
        """Normalize user-provided/legacy query type aliases to canonical names."""
        mapping = {
            "factual": "fact_seeking",
            "facts": "fact_seeking",
            "concept": "concept_seeking",
            "concepts": "concept_seeking",
            "web": "web_search_like",
            "web_like": "web_search_like",
            "exact": "exact_snippet",
            "exact_snippet_search": "exact_snippet",
            "misspelling": "misspellings",
            "spelling_errors": "misspellings",
        }
        key = (qt or "").strip().lower()
        return mapping.get(key, key)

    def _postprocess_query(self, raw: str, query_type: str, bundle: SelectionBundle) -> str:
        """Light cleanup and enforcement of simple constraints per type."""
        text = (raw or "").strip()
        # Strip quotes and trailing punctuation-only lines
        text = re.sub(r'^[\"\'\s]+|[\"\'\s]+$', "", text)
        text = re.sub(r"\s+", " ", text)

        def token_count(s: str) -> int:
            return len(re.findall(r"[A-Za-z0-9]+", s))

        if query_type == "keyword":
            # Keep top 1-4 content words
            tokens = re.findall(r"[A-Za-z0-9]+", text)
            text = " ".join(tokens[:4])
        elif query_type == "short":
            # Trim to <= 4 tokens
            toks = re.findall(r"[A-Za-z0-9]+", text)
            if len(toks) > 4:
                text = " ".join(toks[:4])
        elif query_type == "medium":
            # Force into 5-20 tokens if too long
            toks = re.findall(r"[A-Za-z0-9]+", text)
            if len(toks) > 20:
                text = " ".join(toks[:20])
        elif query_type == "long":
            # Ensure at least 21 tokens by padding from context if too short
            if token_count(text) < 21:
                extra = " ".join([c.chunk_text for c in bundle.golden_chunks])
                extra_toks = re.findall(r"[A-Za-z0-9]+", extra)
                need = 21 - token_count(text)
                text = (text + " " + " ".join(extra_toks[:max(0, need)])).strip()
        elif query_type == "misspellings":
            # Ensure at least one typo: simple vowel drop/transposition
            if not re.search(r"\d|\b[a-z]{2,}[^a-z\s][a-z]*\b", text, flags=re.I):
                words = re.findall(r"[A-Za-z]+", text)
                if words:
                    w = random.choice(words)
                    if len(w) > 3:
                        i = random.randrange(1, len(w)-1)
                        typo = w[:i] + w[i+1] + w[i] + w[i+2:]
                    else:
                        typo = w[:-1]
                    text = re.sub(rf"\b{re.escape(w)}\b", typo, text, count=1)
        elif query_type == "exact_snippet":
            # If not a substring, fallback to selecting a contiguous substring from golden context
            golden_text = "\n".join([c.chunk_text for c in bundle.golden_chunks])
            if text and text in golden_text:
                pass
            else:
                # Select a reasonably long sentence/piece
                sentences = re.split(r"(?<=[\.!?])\s+", golden_text)
                pick = max(sentences, key=lambda s: len(s)) if sentences else golden_text
                # Clip length
                pick = pick.strip()
                if len(pick) > 160:
                    pick = pick[:160].rsplit(" ", 1)[0]
                text = pick

        # Final cleanup
        return text.strip()

    def _get_cached_query(self, bundle: SelectionBundle, query_type: str) -> str:
        """Try to get a cached query for this bundle and query type."""
        if not self.cache:
            return None
            
        # Create cache key from bundle content and config
        golden_text = " ".join([c.chunk_text for c in bundle.golden_chunks])
        distractor_texts = [c.chunk_text for c in bundle.distractor_chunks]
        
        temperature = getattr(self.config, "TEMPERATURE", 0.5)
        max_tokens = getattr(self.config, "MAX_TOKENS", 32)
        
        cache_key = create_prompt_cache_key(
            golden_text, distractor_texts, query_type, 
            temperature, max_tokens
        )
        
        # Add config hash to ensure cache invalidation on config changes
        full_cache_key = f"{cache_key}_{self.config_hash}"
        
        cached_result = self.cache.get(full_cache_key)
        if cached_result:
            logger.debug("Cache hit for query type %s", query_type)
            return cached_result
        
        return None
    
    def _cache_query(self, bundle: SelectionBundle, query_type: str, query_text: str):
        """Cache a generated query for future use."""
        if not self.cache:
            return
            
        # Create same cache key as _get_cached_query
        golden_text = " ".join([c.chunk_text for c in bundle.golden_chunks])
        distractor_texts = [c.chunk_text for c in bundle.distractor_chunks]
        
        temperature = getattr(self.config, "TEMPERATURE", 0.5)
        max_tokens = getattr(self.config, "MAX_TOKENS", 32)
        
        cache_key = create_prompt_cache_key(
            golden_text, distractor_texts, query_type, 
            temperature, max_tokens
        )
        
        # Add config hash to ensure cache invalidation on config changes
        full_cache_key = f"{cache_key}_{self.config_hash}"
        
        self.cache.set(full_cache_key, query_text)
        logger.debug("Cached query for type %s with key %s", query_type, cache_key[:8])
    
    def get_cache_stats(self):
        """Get cache statistics for monitoring."""
        if not self.cache:
            return {"caching": "disabled"}
        return self.cache.size_info()

# --- End File: search-evaluation-api/generation/query_generator.py ---

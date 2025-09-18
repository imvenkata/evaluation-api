#!/usr/bin/env python3
"""
Examples of how caching will work in your pipeline.
These are practical examples you can test.
"""

import sys
sys.path.append('.')

from evaluation_api.utils.cache_utils import SimpleCache, cached, create_prompt_cache_key
import time

# Example 1: Simple caching with SimpleCache class
def example_data_loading_cache():
    """Example: Cache expensive data loading operations."""
    print("🗄️  Example 1: Data Loading Cache")
    
    cache = SimpleCache(cache_dir="./cache", namespace="data_loading")
    
    def expensive_data_load(file_path: str, config_hash: str):
        """Simulate expensive data loading."""
        print(f"    📁 Loading data from {file_path}...")
        time.sleep(2)  # Simulate slow loading
        return {
            "chunks": [f"chunk_{i}" for i in range(100)],
            "metadata": {"file": file_path, "timestamp": time.time()}
        }
    
    # First call - cache miss
    print("  First call (cache miss):")
    start = time.time()
    result1 = cache.get_or_compute(expensive_data_load, "/path/to/data.jsonl", "config_v1")
    print(f"    ⏱️  Took {time.time() - start:.2f} seconds")
    print(f"    📊 Loaded {len(result1['chunks'])} chunks")
    
    # Second call - cache hit!
    print("  Second call (cache hit):")
    start = time.time()
    result2 = cache.get_or_compute(expensive_data_load, "/path/to/data.jsonl", "config_v1")
    print(f"    ⏱️  Took {time.time() - start:.2f} seconds")
    print(f"    📊 Loaded {len(result2['chunks'])} chunks")
    
    print(f"  💾 Cache info: {cache.size_info()}")
    print()


# Example 2: Using the @cached decorator
@cached(cache_dir="./cache", namespace="validation")
def expensive_validation(chunk_text: str, min_tokens: int, max_tokens: int):
    """Simulate expensive chunk validation."""
    print(f"    🔍 Validating chunk: '{chunk_text[:30]}...'")
    time.sleep(0.5)  # Simulate processing time
    
    token_count = len(chunk_text.split())
    is_valid = min_tokens <= token_count <= max_tokens
    
    return {
        "is_valid": is_valid,
        "token_count": token_count,
        "reason": "Valid" if is_valid else f"Token count {token_count} out of range"
    }

def example_validation_cache():
    """Example: Cache chunk validation results."""
    print("✅ Example 2: Validation Cache (using @cached decorator)")
    
    chunks = [
        "This is a short chunk.",
        "This is a much longer chunk with many more words and tokens for testing purposes.",
        "Short again.",
        "This is a short chunk."  # Duplicate - should be cached
    ]
    
    for i, chunk in enumerate(chunks):
        print(f"  Validating chunk {i+1}:")
        start = time.time()
        result = expensive_validation(chunk, 3, 20)
        print(f"    ⏱️  Took {time.time() - start:.2f} seconds")
        print(f"    ✅ Result: {result['reason']}")
    
    print(f"  💾 Cache info: {expensive_validation.cache_info()}")
    print()


# Example 3: LLM Query Generation Cache (most important!)
def example_llm_cache():
    """Example: Cache expensive LLM query generation."""
    print("🤖 Example 3: LLM Query Generation Cache")
    
    cache = SimpleCache(cache_dir="./cache", namespace="llm_queries")
    
    def simulate_llm_call(golden_text: str, distractor_texts: list, query_type: str):
        """Simulate expensive Azure OpenAI call."""
        cache_key = create_prompt_cache_key(
            golden_text, distractor_texts, query_type, 
            temperature=0.5, max_tokens=32
        )
        print(f"    🔑 Cache key: {cache_key}")
        print(f"    🤖 Calling Azure OpenAI for {query_type} query...")
        time.sleep(1)  # Simulate API call
        
        # Simulate generated query
        return f"What is {query_type} information about {golden_text.split()[0]}?"
    
    golden = "Machine learning algorithms are used for pattern recognition."
    distractors = ["Deep learning models process data.", "AI systems learn patterns."]
    
    # Test multiple query types
    query_types = ["factual", "keyword", "factual"]  # Note: "factual" appears twice
    
    for i, qtype in enumerate(query_types):
        print(f"  Query {i+1} ({qtype}):")
        start = time.time()
        
        # Check cache first
        cache_key = create_prompt_cache_key(golden, distractors, qtype, 0.5, 32)
        cached_result = cache.get(cache_key)
        
        if cached_result:
            print(f"    💰 Cache hit! Retrieved in {time.time() - start:.3f} seconds")
            query = cached_result
        else:
            query = simulate_llm_call(golden, distractors, qtype)
            cache.set(cache_key, query)
            print(f"    💸 Cache miss. Generated in {time.time() - start:.2f} seconds")
        
        print(f"    📝 Query: '{query}'")
    
    print(f"  💾 Cache info: {cache.size_info()}")
    print()


# Example 4: Cache invalidation
def example_cache_invalidation():
    """Example: How cache handles configuration changes."""
    print("🔄 Example 4: Cache Invalidation")
    
    cache = SimpleCache(cache_dir="./cache", namespace="config_test")
    
    def process_with_config(text: str, temperature: float):
        print(f"    ⚙️  Processing with temperature={temperature}")
        time.sleep(0.5)
        return f"Processed '{text}' with temp {temperature}"
    
    text = "Sample text for processing"
    
    # First call with temperature 0.5
    print("  Processing with temperature=0.5:")
    result1 = cache.get_or_compute(process_with_config, text, 0.5)
    print(f"    📤 Result: {result1}")
    
    # Same call - cache hit
    print("  Same call again (cache hit):")
    result2 = cache.get_or_compute(process_with_config, text, 0.5)
    print(f"    📤 Result: {result2}")
    
    # Different temperature - cache miss (different key)
    print("  Processing with temperature=0.7 (different key):")
    result3 = cache.get_or_compute(process_with_config, text, 0.7)
    print(f"    📤 Result: {result3}")
    
    print(f"  💾 Cache info: {cache.size_info()}")
    print()


def main():
    """Run all cache examples."""
    print("🚀 Cache System Examples")
    print("=" * 50)
    print("This demonstrates how caching will work in your pipeline.\n")
    
    # Run examples
    example_data_loading_cache()
    example_validation_cache()
    example_llm_cache()
    example_cache_invalidation()
    
    print("✅ All examples completed!")
    print("\n💡 Key Takeaways:")
    print("   • First run: Cache miss (slow)")
    print("   • Subsequent runs: Cache hit (fast)")
    print("   • Different inputs: New cache entries")
    print("   • Same inputs: Instant cache retrieval")
    print("   • Configuration changes: Automatic cache invalidation")

if __name__ == "__main__":
    main()

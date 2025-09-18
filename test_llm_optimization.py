#!/usr/bin/env python3
"""
Test script to validate LLM rate limit optimizations.
Run this to verify your settings before processing the full dataset.
"""

import os
import time
import sys
sys.path.append('.')

from evaluation_api.configs import generation_config
from evaluation_api.generation.query_generator import QueryGenerator
from evaluation_api.generation.models import SelectionBundle, ChunkData

def test_llm_optimization():
    """Test the optimized LLM settings with a small sample."""
    
    print("🚀 Testing LLM Rate Limit Optimization")
    print("=" * 50)
    
    # Display current settings
    print(f"LLM_MAX_WORKERS: {getattr(generation_config, 'LLM_MAX_WORKERS', 'Not set')}")
    print(f"LLM_MAX_QPS: {getattr(generation_config, 'LLM_MAX_QPS', 'Not set')}")
    print(f"TEMPERATURE: {getattr(generation_config, 'TEMPERATURE', 'Not set')}")
    print(f"MAX_TOKENS: {getattr(generation_config, 'MAX_TOKENS', 'Not set')}")
    print(f"LLM_BURST_CAPACITY: {getattr(generation_config, 'LLM_BURST_CAPACITY', 'Not set')}")
    print()
    
    # Check Azure OpenAI configuration
    if not hasattr(generation_config, 'AZURE_OPENAI_ENDPOINT') or not generation_config.AZURE_OPENAI_ENDPOINT:
        print("❌ AZURE_OPENAI_ENDPOINT not configured")
        print("Please set your Azure OpenAI endpoint in the config or environment variables")
        return False
    
    if not os.environ.get("AZURE_OPENAI_KEY"):
        print("❌ AZURE_OPENAI_KEY not found in environment variables")
        print("Please set: export AZURE_OPENAI_KEY='your-key-here'")
        return False
    
    print("✅ Azure OpenAI configuration found")
    
    # Create test data
    test_chunks = [
        ChunkData(
            doc_id="test_doc_1",
            chunk_id="test_chunk_1", 
            chunk_text="This is a test chunk about machine learning algorithms.",
            embedding=[0.1] * 1024
        ),
        ChunkData(
            doc_id="test_doc_2", 
            chunk_id="test_chunk_2",
            chunk_text="This chunk discusses natural language processing techniques.",
            embedding=[0.2] * 1024
        )
    ]
    
    # Create test bundles
    test_bundles = [
        SelectionBundle(
            golden_chunks=[test_chunks[0]],
            distractor_chunks=[test_chunks[1]]
        )
    ]
    
    print(f"📝 Testing with {len(test_bundles)} bundles...")
    
    try:
        # Initialize query generator
        generator = QueryGenerator(generation_config)
        
        # Test query generation
        start_time = time.time()
        queries = generator.generate_queries(test_bundles)
        end_time = time.time()
        
        # Results
        print(f"✅ Generated {len(queries)} queries in {end_time - start_time:.2f} seconds")
        
        if queries:
            print(f"📄 Sample query: '{queries[0].query}'")
            
        # Calculate theoretical performance
        estimated_qps = len(queries) / (end_time - start_time) if end_time > start_time else 0
        queries_per_50k = 50000 * len(generation_config.QUERY_TYPES)
        estimated_hours = queries_per_50k / (estimated_qps * 3600) if estimated_qps > 0 else float('inf')
        
        print(f"📊 Estimated QPS: {estimated_qps:.2f}")
        print(f"📊 Estimated time for 50k chunks: {estimated_hours:.1f} hours")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your Azure OpenAI quota limits")
        print("2. Try reducing LLM_MAX_QPS")
        print("3. Verify your API key and endpoint")
        return False

def main():
    """Main test function."""
    print("Testing LLM optimization settings...")
    print("This will make a few test API calls to validate your configuration.\n")
    
    success = test_llm_optimization()
    
    if success:
        print("\n🎉 Optimization test successful!")
        print("You can now run the full pipeline with improved performance.")
        print("\nNext steps:")
        print("1. Monitor Azure Portal for usage")
        print("2. Adjust LLM_MAX_QPS based on your quota")
        print("3. Run with SELECTION_SAMPLE_RATE=0.01 for initial testing")
    else:
        print("\n❌ Optimization test failed.")
        print("Please review the configuration and try again.")

if __name__ == "__main__":
    main()

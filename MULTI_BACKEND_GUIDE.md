# Multi-Backend Search Guide

This guide explains how to use the configurable search backends in your synthetic ground truth generation pipeline.

## 🏗️ **Architecture Overview**

The system now supports multiple search backends through a pluggable architecture:

```text
┌─────────────────────────────────────────┐
│           ContextSelector               │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐    │
│  │     SearchBackend (ABC)         │    │
│  │  + find_similar_chunks()        │    │
│  │  + find_duplicates()            │    │
│  │  + initialize()                 │    │
│  └─────────────────────────────────┘    │
│               │                         │
│    ┌──────────┼──────────┐              │
│    ▼          ▼          ▼              │
│ ┌─────────┐ ┌──────────┐ ┌────────────┐ │
│ │  FAISS  │ │  Azure   │ │   Hybrid   │ │
│ │Backend  │ │ Search   │ │  Search    │ │
│ └─────────┘ └──────────┘ └────────────┘ │
└─────────────────────────────────────────┘
```

## 🎯 **Available Backends**

### **1. FAISS Backend** (`faiss`)
- **Best for**: Small to medium datasets (< 100k chunks), teams without Azure AI Search
- **Pros**: Fast, no external dependencies, works offline
- **Cons**: Memory intensive, requires manual optimization
- **Memory**: ~6-8GB for 500k chunks

### **2. Azure AI Search Backend** (`azure_search`)
- **Best for**: Large datasets (100k+ chunks), teams with existing Azure AI Search
- **Pros**: Scales to millions, minimal memory usage, production-ready
- **Cons**: Requires API calls, costs per request
- **Memory**: <100MB regardless of dataset size

### **3. Hybrid Search Backend** (`hybrid_search`)
- **Best for**: Maximum distractor quality, sophisticated evaluation scenarios
- **Pros**: Combines vector + text search, highest quality results
- **Cons**: More API calls, potentially slower
- **Memory**: <100MB + API latency

## ⚙️ **Configuration**

### **Basic Configuration**

Set the search backend in your `generation_config.py`:

```python
# Choose your backend
SEARCH_BACKEND = "faiss"          # Default: FAISS
# SEARCH_BACKEND = "azure_search"   # Azure AI Search
# SEARCH_BACKEND = "hybrid_search"  # Hybrid Azure Search
```

### **Environment Variables**

You can also set via environment variables:

```bash
export SEARCH_BACKEND=azure_search
```

### **FAISS Backend Settings**

```python
# FAISS-specific configuration
USE_IVF_SELECTION = True          # Use IVF for large datasets
IVF_NLIST = 512                   # Number of clusters
IVF_NPROBE = 32                   # Search probes
DEDUP_MAX_NEIGHBORS = 20          # Max neighbors for dedup
```

### **Azure Search Backend Settings**

```python
# Azure AI Search configuration
AZURE_SEARCH_ENDPOINT = "https://your-service.search.windows.net"
AZURE_SEARCH_INDEX_NAME = "your-index-name"
AZURE_SEARCH_KEY = "your-api-key"

# Field mappings (adjust to match your index schema)
AZURE_SEARCH_VECTOR_FIELD = "content_vector"    # Embedding field
AZURE_SEARCH_CONTENT_FIELD = "chunk_text"       # Text content field
AZURE_SEARCH_DOC_ID_FIELD = "doc_id"           # Document ID field
AZURE_SEARCH_CHUNK_ID_FIELD = "chunk_id"       # Chunk ID field

# Performance tuning
AZURE_SEARCH_BATCH_SIZE = 100                   # Batch requests
AZURE_SEARCH_RETRY_ATTEMPTS = 3                # Retry failed requests
AZURE_SEARCH_TIMEOUT_SECONDS = 30              # Request timeout
```

### **Environment Variables for Azure Search**

```bash
export AZURE_SEARCH_ENDPOINT="https://your-service.search.windows.net"
export AZURE_SEARCH_INDEX_NAME="your-index-name"
export AZURE_SEARCH_KEY="your-api-key"
export SEARCH_BACKEND="azure_search"
```

## 🚀 **Usage Examples**

### **Example 1: Small Dataset with FAISS**

```python
# generation_config.py
SEARCH_BACKEND = "faiss"
INPUT_PATHS = ["/path/to/small/dataset"]  # < 10k chunks
SELECTION_SAMPLE_RATE = 1.0
NUM_DISTRACTORS = 3
```

```bash
python -m evaluation_api.generation.cli --config /path/to/generation_config.py
```

### **Example 2: Large Dataset with Azure AI Search**

```python
# generation_config.py
SEARCH_BACKEND = "azure_search"
INPUT_PATHS = ["/path/to/large/dataset"]  # 500k chunks
SELECTION_SAMPLE_RATE = 0.1  # Sample 10% for efficiency
NUM_DISTRACTORS = 5

# Azure Search settings
AZURE_SEARCH_ENDPOINT = "https://myservice.search.windows.net"
AZURE_SEARCH_INDEX_NAME = "document-chunks"
AZURE_SEARCH_KEY = "your-key-here"
```

```bash
export AZURE_SEARCH_KEY="your-actual-key"
python -m evaluation_api.generation.cli --config /path/to/generation_config.py
```

### **Example 3: Hybrid Search for Maximum Quality**

```python
# generation_config.py
SEARCH_BACKEND = "hybrid_search"
INPUT_PATHS = ["/path/to/dataset"]
SELECTION_SAMPLE_RATE = 0.05  # Lower sample for cost efficiency
NUM_DISTRACTORS = 8  # More distractors for better evaluation

# Hybrid search finds both semantically and lexically similar content
```

## 📊 **Performance Comparison**

| Backend | Memory Usage | Setup Time | Query Speed | Scalability | Cost |
|---------|--------------|------------|-------------|-------------|------|
| FAISS | High (6-8GB) | Medium (5-15min) | Very Fast (<1ms) | Limited | Free |
| Azure Search | Low (<100MB) | None | Fast (10-50ms) | Excellent | API calls |
| Hybrid | Low (<100MB) | None | Medium (20-100ms) | Excellent | More API calls |

## 🔧 **Advanced Configuration**

### **Dynamic Backend Selection**

You can choose backends based on dataset size:

```python
# Auto-select backend based on data size
def choose_backend(num_chunks):
    if num_chunks < 10000:
        return "faiss"
    elif num_chunks < 100000:
        return "azure_search"
    else:
        return "hybrid_search"

SEARCH_BACKEND = choose_backend(len(load_data_preview()))
```

### **Backend-Specific Optimizations**

```python
# FAISS optimizations for large datasets
if SEARCH_BACKEND == "faiss":
    USE_IVF_SELECTION = True
    IVF_NLIST = min(4096, max(256, int(sqrt(num_chunks))))

# Azure Search optimizations
elif SEARCH_BACKEND in ["azure_search", "hybrid_search"]:
    AZURE_SEARCH_BATCH_SIZE = 200
    SELECTION_SAMPLE_RATE = 0.1  # Reduce API calls
```

### **Fallback Strategy**

```python
# Fallback configuration
USE_BACKEND_DUPLICATE_DETECTION = True  # Try backend first
# If backend fails, automatically falls back to legacy FAISS/cosine method
```

## 🐛 **Troubleshooting**

### **FAISS Issues**

```bash
# Install FAISS
pip install faiss-cpu

# For GPU support
pip install faiss-gpu
```

### **Azure Search Issues**

```bash
# Install Azure Search SDK
pip install azure-search-documents>=11.6.0

# Test connection
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

client = SearchClient(
    endpoint="https://your-service.search.windows.net",
    index_name="your-index",
    credential=AzureKeyCredential("your-key")
)

# Test query
results = list(client.search("test", top=1))
print(f"Connection successful: {len(results)} results")
```

### **Common Error Messages**

| Error | Solution |
|-------|----------|
| "Failed to initialize search backend" | Check configuration and dependencies |
| "Azure Search client not initialized" | Verify endpoint, index name, and API key |
| "faiss not installed" | Run `pip install faiss-cpu` |
| "Inconsistent embedding dimension" | Ensure all embeddings have same dimension |

## 📈 **Migration Guide**

### **From Pure FAISS to Multi-Backend**

1. **No changes needed** - FAISS remains the default
2. **Optional**: Set `SEARCH_BACKEND = "faiss"` explicitly
3. **Test**: Run existing pipelines to ensure compatibility

### **Adding Azure Search to Existing Setup**

1. **Configure Azure Search** in `generation_config.py`
2. **Set environment variables** for credentials
3. **Change backend**: `SEARCH_BACKEND = "azure_search"`
4. **Test with small sample** first
5. **Scale up** once verified

## 💡 **Best Practices**

### **Backend Selection Strategy**

```python
# Recommended backend selection
def recommend_backend(num_chunks, has_azure_search, budget_conscious):
    if num_chunks < 50000 and not budget_conscious:
        return "faiss"  # Fast and free
    elif has_azure_search and num_chunks > 100000:
        return "hybrid_search"  # Best quality
    elif has_azure_search:
        return "azure_search"  # Good balance
    else:
        return "faiss"  # Fallback
```

### **Cost Optimization for Azure Search**

```python
# Reduce costs with smart sampling
SELECTION_SAMPLE_RATE = 0.05  # Process 5% of chunks
NUM_DISTRACTORS = 3           # Fewer API calls per chunk

# Batch processing (future enhancement)
AZURE_SEARCH_BATCH_SIZE = 100
```

### **Quality vs Performance Trade-offs**

```python
# High quality (slower, more expensive)
SEARCH_BACKEND = "hybrid_search"
NUM_DISTRACTORS = 10
SELECTION_SAMPLE_RATE = 0.2

# Balanced (good quality, reasonable cost)
SEARCH_BACKEND = "azure_search"
NUM_DISTRACTORS = 5
SELECTION_SAMPLE_RATE = 0.1

# Fast processing (lower quality)
SEARCH_BACKEND = "faiss"
NUM_DISTRACTORS = 3
SELECTION_SAMPLE_RATE = 1.0
```

## 🔄 **Future Enhancements**

Planned improvements to the multi-backend system:

1. **Elasticsearch Backend** - For teams using Elasticsearch
2. **Pinecone Backend** - For vector database users
3. **Batch Processing** - Reduce API calls with batching
4. **Caching Layer** - Cache similarity results
5. **Auto-Backend Selection** - Choose based on data characteristics
6. **Performance Monitoring** - Track backend performance metrics

## 📞 **Support**

For issues with specific backends:

- **FAISS**: Check memory usage and dataset size
- **Azure Search**: Verify API quotas and index schema
- **General**: Review logs for backend initialization messages

The system will log which backend is being used:
```
INFO [chunk_selector] - Using search backend: {'backend': 'Azure AI Search', 'endpoint': '...', 'num_chunks': 500000}
```

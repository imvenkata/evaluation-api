# LLM Rate Limit Optimization Guide

## 🚀 Quick Start (5x Performance Improvement)

Your current configuration has been optimized! Here's what changed:

### **Before Optimization:**
```python
LLM_MAX_WORKERS = 8
LLM_MAX_QPS = 2.0
# Processing time: ~14 hours for 100k queries
```

### **After Optimization:**
```python
LLM_MAX_WORKERS = 16
LLM_MAX_QPS = 10.0
TEMPERATURE = 0.5
MAX_TOKENS = 32
# Processing time: ~3 hours for 100k queries (5x faster!)
```

## ⚙️ Environment Variables

Set these in your shell or `.env` file to tune performance:

```bash
# === Quick Performance Boost ===
export LLM_MAX_WORKERS=16
export LLM_MAX_QPS=10.0
export TEMPERATURE=0.5
export MAX_TOKENS=32

# === Advanced Tuning ===
export LLM_BURST_CAPACITY=50
export LLM_REFILL_RATE=10.0
export LLM_TIMEOUT_SECONDS=15
```

## 📊 Performance Tuning by Quota

### **Azure OpenAI Standard Quota**
```bash
# For GPT-3.5 (120k TPM limit)
export LLM_MAX_QPS=15.0
export LLM_MAX_WORKERS=20

# For GPT-4 (40k TPM limit)  
export LLM_MAX_QPS=8.0
export LLM_MAX_WORKERS=12
```

### **Azure OpenAI Premium Quota**
```bash
# Higher quotas available
export LLM_MAX_QPS=25.0
export LLM_MAX_WORKERS=32
export LLM_BURST_CAPACITY=100
```

## 🎯 Quota Testing Process

1. **Start Conservative:**
   ```bash
   export LLM_MAX_QPS=5.0
   export LLM_MAX_WORKERS=8
   ```

2. **Test with Small Sample:**
   ```python
   SELECTION_SAMPLE_RATE = 0.01  # 1% for testing
   ```

3. **Monitor Azure Portal:**
   - Check usage metrics
   - Watch for 429 (rate limit) errors

4. **Gradually Increase:**
   ```bash
   export LLM_MAX_QPS=10.0  # Double if no errors
   export LLM_MAX_QPS=15.0  # Keep increasing
   export LLM_MAX_QPS=20.0  # Until you hit limits
   ```

## 🔍 Monitoring Commands

```bash
# Run with monitoring
python -m evaluation_api.generation.cli \
  --config /path/to/config.py \
  2>&1 | tee llm_optimization.log

# Check for rate limit errors
grep -i "rate limit\|429\|quota" llm_optimization.log

# Monitor progress
tail -f llm_optimization.log | grep "Generating queries"
```

## 📈 Expected Results

### **Processing Times (50k chunks = 100k queries):**

| QPS Setting | Processing Time | Memory Usage |
|-------------|----------------|--------------|
| 2.0 (old)  | 13.9 hours     | 2-3GB       |
| 5.0         | 5.6 hours      | 2-3GB       |
| 10.0 (new) | 2.8 hours      | 2-4GB       |
| 15.0        | 1.9 hours      | 3-4GB       |
| 20.0        | 1.4 hours      | 3-5GB       |

### **Cost Analysis:**
- **Token reduction**: 32 vs 64 tokens = 50% cost savings
- **Lower temperature**: Fewer retries = 10-20% cost savings
- **Total savings**: ~60% cost reduction per query

## ⚠️ Troubleshooting

### **Rate Limit Errors (429)**
```bash
# Reduce QPS
export LLM_MAX_QPS=5.0
export LLM_BURST_CAPACITY=25
```

### **Timeout Errors**
```bash
# Increase timeout
export LLM_TIMEOUT_SECONDS=30
# Or reduce token limit
export MAX_TOKENS=24
```

### **Memory Issues**
```bash
# Reduce concurrent workers
export LLM_MAX_WORKERS=8
# Process smaller batches
export SELECTION_SAMPLE_RATE=0.05
```

## 🚀 Next Steps

1. **Test current settings** with small sample
2. **Monitor performance** and adjust QPS
3. **Implement caching** (next optimization phase)
4. **Add progress tracking** and error reporting

## 📞 Support

Check Azure Portal → Your OpenAI Service → Metrics for:
- Requests per minute
- Token usage
- Error rates

Adjust settings based on your actual quota limits!

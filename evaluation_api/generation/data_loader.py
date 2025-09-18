# --- File: search-evaluation-api/generation/data_loader.py ---
# This module implements the logic for loading data based on the config.

import json
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List
from .models import ChunkData
from ..utils.cache_utils import SimpleCache

# Optional fast JSON parser
try:
    import importlib
    orjson = importlib.import_module("orjson")  # type: ignore
except (ModuleNotFoundError, ImportError):
    orjson = None  # type: ignore

logger = logging.getLogger(__name__)

def load_data(config) -> List[ChunkData]:
    """
    Loads data based on the INPUT_TYPE specified in the config.
    Uses caching to avoid reloading unchanged data.
    """
    # Initialize cache if enabled
    cache = None
    cache_enabled = getattr(config, 'CACHE_DATA_LOADING', True) and getattr(config, 'ENABLE_CACHING', True)
    
    if cache_enabled:
        cache_dir = getattr(config, 'CACHE_DIR', './cache')
        cache = SimpleCache(cache_dir, namespace="data_loading")
    
    if config.INPUT_TYPE == "chunks":
        return _load_from_chunks(config.INPUT_PATHS, config, cache)
    elif config.INPUT_TYPE == "documents":
        raise NotImplementedError(
            "Document processing is not yet implemented. "
            "Please set INPUT_TYPE to 'chunks' in your config."
        )
    elif config.INPUT_TYPE == "azure_blob_chunks":
        return _load_from_azure_blob(config, cache)
    else:
        raise ValueError(f"Unknown INPUT_TYPE: {config.INPUT_TYPE}")

def _load_from_chunks(input_paths: List[str], config, cache=None) -> List[ChunkData]:
    """Loads pre-computed chunks from JSONL files or directories of JSON/JSONL files with caching."""
    chunks = []
    dims_seen = set()
    
    for path in input_paths:
        # Try cache first if available
        if cache:
            cached_chunks = _try_load_cached_path(path, cache)
            if cached_chunks is not None:
                logger.info("Loading cached chunks from %s (%d chunks)", path, len(cached_chunks))
                chunks.extend(cached_chunks)
                # Update dims_seen from cached chunks
                for chunk in cached_chunks:
                    if chunk.embedding:
                        dims_seen.add(len(chunk.embedding))
                continue
        
        logger.info("Loading pre-computed chunks from %s...", path)
        if os.path.isdir(path):
            # Collect JSONL and JSON files in directory
            dir_files = sorted(
                [p for p in glob(os.path.join(path, "*.jsonl"))] +
                [p for p in glob(os.path.join(path, "*.json"))]
            )
            if not dir_files:
                logger.warning("No .json/.jsonl files found in directory: %s", path)
            file_paths = dir_files
        else:
            file_paths = [path]

        # Load and cache this path's chunks
        path_chunks = []
        for fp in file_paths:
            try:
                if fp.endswith('.jsonl'):
                    with open(fp, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                if orjson is not None:
                                    data = orjson.loads(line)
                                else:
                                    data = json.loads(line)
                                chunk_obj, dim = _to_chunkdata(data)
                                path_chunks.append(chunk_obj)
                                if dim is not None:
                                    dims_seen.add(dim)
                            except (json.JSONDecodeError, KeyError, ValueError) as e:
                                logger.warning("Skipping malformed line in %s: %s", fp, e)
                elif fp.endswith('.json'):
                    with open(fp, 'r', encoding='utf-8') as f:
                        try:
                            if orjson is not None:
                                data = orjson.loads(f.read())
                            else:
                                data = json.load(f)
                            chunk_obj, dim = _to_chunkdata(data)
                            path_chunks.append(chunk_obj)
                            if dim is not None:
                                dims_seen.add(dim)
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.warning("Skipping malformed JSON in %s: %s", fp, e)
                else:
                    logger.warning("Unsupported file type (skipped): %s", fp)
            except FileNotFoundError:
                logger.error("Input file not found: %s", fp)
                raise
        
        # Cache the loaded chunks for this path
        chunks.extend(path_chunks)
        if cache and path_chunks:
            _cache_path_chunks(path, path_chunks, cache)
    
    if not chunks:
        logger.error("No chunks were loaded. Please check INPUT_PATHS.")
    # Consistency checks for embedding dimensions
    if len(dims_seen) > 1:
        logger.error("Inconsistent embedding dimensions detected: %s", sorted(dims_seen))
        raise ValueError("Inconsistent embedding dimensions across loaded chunks.")
    if dims_seen:
        dim = list(dims_seen)[0]
        expected = getattr(config, 'EMBED_DIM', None)
        if expected is not None and dim != expected:
            logger.error("Loaded embedding dim %s does not match config.EMBED_DIM=%s", dim, expected)
            raise ValueError(f"Embedding dimension mismatch: data={dim}, config={expected}")

    return chunks

def _to_chunkdata(data: dict):
    doc_id = data['doc_id']
    chunk_id = data['chunk_id']
    chunk_text = data.get('chunk_text', data.get('chunk', ''))
    embedding = data.get('embedding', data.get('content_vector'))
    if embedding is None:
        raise KeyError("Missing 'embedding' or 'content_vector' in data")
    # Ensure list of floats
    embedding = [float(x) for x in embedding]
    return ChunkData(
        doc_id=doc_id,
        chunk_id=chunk_id,
        chunk_text=chunk_text,
        embedding=embedding
    ), len(embedding)


def _load_from_azure_blob(config, cache=None) -> List[ChunkData]:
    """Loads per-file JSON chunks from Azure Blob Storage concurrently.
    Expects container + prefix to locate many small .json files.
    
    Note: Azure Blob caching implementation is planned for future enhancement.
    """
    # TODO: Implement Azure Blob caching in future version
    _ = cache  # Suppress unused argument warning
    try:
        from azure.storage.blob import ContainerClient  # type: ignore
        from azure.identity import DefaultAzureCredential  # type: ignore
    except ImportError as e:
        logger.error("Azure SDK not installed: %s", e)
        raise

    account_url = os.getenv("AZURE_BLOB_ACCOUNT_URL", getattr(config, "BLOB_ACCOUNT_URL", ""))
    container = getattr(config, "BLOB_CONTAINER", None)
    prefix = getattr(config, "BLOB_PREFIX", "")
    max_workers = int(getattr(config, "BLOB_MAX_WORKERS", 64))
    if not account_url or not container:
        raise ValueError("BLOB_ACCOUNT_URL/AZURE_BLOB_ACCOUNT_URL and BLOB_CONTAINER must be set for azure_blob_chunks")

    # Auth: prefer DefaultAzureCredential, fallback to connection string or SAS via env if present
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
    client = ContainerClient(account_url=account_url, container_name=container, credential=credential)

    logger.info("Listing blobs with prefix: %s", prefix)
    blob_paths = [b.name for b in client.list_blobs(name_starts_with=prefix) if b.name.endswith('.json')]
    if not blob_paths:
        logger.error("No JSON blobs found under prefix: %s", prefix)
        return []
    if len(blob_paths) > 50000:
        logger.warning("Large number of small blobs detected: %s. Consider consolidating into larger JSONL files for performance.", len(blob_paths))

    chunks: List[ChunkData] = []
    dims_seen = set()

    def fetch_and_parse(blob_name: str):
        try:
            # Lazy import to avoid hard dependency if not used
            from tenacity import retry, stop_after_attempt, wait_exponential_jitter  # type: ignore
        except (ModuleNotFoundError, ImportError):
            retry = None  # type: ignore
        try:
            from azure.core.exceptions import AzureError  # type: ignore
        except (ModuleNotFoundError, ImportError):  # pragma: no cover
            class AzureError(Exception):
                pass

        def _download_once(name: str):
            downloader = client.download_blob(name)
            data_bytes = downloader.readall()
            if orjson is not None:
                return orjson.loads(data_bytes)
            return json.loads(data_bytes)

        if retry is not None:
            @retry(stop=stop_after_attempt(int(getattr(config, 'BLOB_RETRY_ATTEMPTS', 5))),
                   wait=wait_exponential_jitter(initial=0.2, max=8.0))
            def _download_with_retry(name: str):
                return _download_once(name)
            download_fn = _download_with_retry
        else:
            download_fn = _download_once

        try:
            obj = download_fn(blob_name)
            c, dim = _to_chunkdata(obj)
            return c, dim, None
        except (AzureError, ValueError, json.JSONDecodeError) as e:  # type: ignore[attr-defined]
            return None, None, (blob_name, str(e))

    logger.info("Downloading %s blobs with up to %s workers...", len(blob_paths), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_and_parse, name) for name in blob_paths]
        for fut in as_completed(futures):
            c, dim, err = fut.result()
            if err:
                logger.warning("Skipping blob due to error: %s -> %s", err[0], err[1])
                continue
            if c is not None:
                chunks.append(c)
                if dim is not None:
                    dims_seen.add(dim)

    if not chunks:
        logger.error("No chunks were loaded from Azure Blob. Check prefix and container.")
    if len(dims_seen) > 1:
        logger.error("Inconsistent embedding dims from Azure Blob: %s", sorted(dims_seen))
        raise ValueError("Inconsistent embedding dimensions across loaded chunks from Azure Blob.")
    if dims_seen:
        dim = list(dims_seen)[0]
        expected = getattr(config, 'EMBED_DIM', None)
        if expected is not None and dim != expected:
            logger.error("Loaded embedding dim %s does not match config.EMBED_DIM=%s", dim, expected)
            raise ValueError(f"Embedding dimension mismatch: data={dim}, config={expected}")

    return chunks


def _try_load_cached_path(path: str, cache: SimpleCache) -> List[ChunkData]:
    """Try to load chunks for a path from cache."""
    import hashlib
    
    # Create cache key based on path and modification time
    try:
        if os.path.isdir(path):
            # For directories, use the directory path and most recent file modification time
            dir_files = []
            for ext in ['*.json', '*.jsonl']:
                dir_files.extend(glob(os.path.join(path, ext)))
            
            if not dir_files:
                return None
                
            # Get most recent modification time
            latest_mtime = max(os.path.getmtime(f) for f in dir_files)
            cache_key = hashlib.sha256(f"{path}_{latest_mtime}".encode()).hexdigest()[:16]
        else:
            # For single files, use file path and modification time
            if not os.path.exists(path):
                return None
            mtime = os.path.getmtime(path)
            cache_key = hashlib.sha256(f"{path}_{mtime}".encode()).hexdigest()[:16]
        
        return cache.get(cache_key)
    except (OSError, ValueError):
        return None


def _cache_path_chunks(path: str, chunks: List[ChunkData], cache: SimpleCache):
    """Cache chunks for a specific path."""
    import hashlib
    
    try:
        if os.path.isdir(path):
            # For directories, use the directory path and most recent file modification time
            dir_files = []
            for ext in ['*.json', '*.jsonl']:
                dir_files.extend(glob(os.path.join(path, ext)))
            
            if not dir_files:
                return
                
            # Get most recent modification time
            latest_mtime = max(os.path.getmtime(f) for f in dir_files)
            cache_key = hashlib.sha256(f"{path}_{latest_mtime}".encode()).hexdigest()[:16]
        else:
            # For single files, use file path and modification time
            if not os.path.exists(path):
                return
            mtime = os.path.getmtime(path)
            cache_key = hashlib.sha256(f"{path}_{mtime}".encode()).hexdigest()[:16]
        
        cache.set(cache_key, chunks)
        logger.debug("Cached %d chunks for path %s with key %s", len(chunks), path, cache_key[:8])
    except (OSError, ValueError) as e:
        logger.warning("Failed to cache chunks for path %s: %s", path, e)

# --- End File: search-evaluation-api/generation/data_loader.py ---

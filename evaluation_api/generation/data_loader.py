# --- File: search-evaluation-api/generation/data_loader.py ---
# This module implements the logic for loading data based on the config.

import json
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import List
from .models import ChunkData

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
    """
    if config.INPUT_TYPE == "chunks":
        return _load_from_chunks(config.INPUT_PATHS, config)
    elif config.INPUT_TYPE == "documents":
        raise NotImplementedError(
            "Document processing is not yet implemented. "
            "Please set INPUT_TYPE to 'chunks' in your config."
        )
    elif config.INPUT_TYPE == "azure_blob_chunks":
        return _load_from_azure_blob(config)
    else:
        raise ValueError(f"Unknown INPUT_TYPE: {config.INPUT_TYPE}")

def _load_from_chunks(input_paths: List[str], config) -> List[ChunkData]:
    """Loads pre-computed chunks from JSONL files or directories of JSON/JSONL files."""
    chunks = []
    dims_seen = set()
    for path in input_paths:
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
                                chunks.append(chunk_obj)
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
                            chunks.append(chunk_obj)
                            if dim is not None:
                                dims_seen.add(dim)
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            logger.warning("Skipping malformed JSON in %s: %s", fp, e)
                else:
                    logger.warning("Unsupported file type (skipped): %s", fp)
            except FileNotFoundError:
                logger.error("Input file not found: %s", fp)
                raise
    
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
            logger.warning("Loaded embedding dim %s does not match config.EMBED_DIM=%s", dim, expected)

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


def _load_from_azure_blob(config) -> List[ChunkData]:
    """Loads per-file JSON chunks from Azure Blob Storage concurrently.
    Expects container + prefix to locate many small .json files.
    """
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

    chunks: List[ChunkData] = []
    dims_seen = set()

    def fetch_and_parse(blob_name: str):
        try:
            downloader = client.download_blob(blob_name)
            data_bytes = downloader.readall()
            if orjson is not None:
                obj = orjson.loads(data_bytes)
            else:
                obj = json.loads(data_bytes)
            c, dim = _to_chunkdata(obj)
            return c, dim, None
        except Exception as e:  # noqa: BLE001
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
            logger.warning("Loaded embedding dim %s does not match config.EMBED_DIM=%s", dim, expected)

    return chunks

# --- End File: search-evaluation-api/generation/data_loader.py ---

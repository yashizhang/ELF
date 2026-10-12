"""Prepare one shared C4 prefix, pretrained snapshot, and released OWT tokens.

Resume is transactional at source shard boundaries. A killed shard may be
retokenized, but no committed prefix is repeated in the resulting dataset.
Document IDs are (revision, source shard, JSONL row); the document index stores
absolute token offsets so every fixed packing boundary can be reconstructed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from .common import (atomic_json, directory_manifest, file_hash, load_config,
                     object_hash, preparation_lock, read_json, sources)

PREPROCESSING_VERSION = "c4-packed-eos-u16-v1"


def model_snapshot(root, name="t5"):
    spec = sources()[name]
    path = Path(root) / name
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["source"] != spec:
            raise ValueError("Snapshot source mismatch")
        return path
    snapshot_download(spec["id"], revision=spec["revision"], local_dir=str(path),
                      allow_patterns=["config.json", "model.safetensors", "tokenizer*",
                                      "spiece.model", "special_tokens_map.json", "added_tokens.json",
                                      "merges.txt", "vocab.json", "generation_config.json"],
                      max_workers=4)
    manifest = directory_manifest(path) | {"source": spec, "status": "complete"}
    atomic_json(manifest_path, manifest)
    return path


def prepare_c4(root, split, examples, source_length, tokenizer, batch_docs=256):
    root = Path(root)
    out = root / "c4" / split
    out.mkdir(parents=True, exist_ok=True)
    spec = sources()["c4"]
    identity = {"source": spec, "split": split, "source_length": source_length,
                "examples": examples, "preprocessing": PREPROCESSING_VERSION,
                "tokenizer": sources()["t5"], "order": "lexicographic source shard, JSONL row, contiguous tokens",
                "packing": "append EOS to each document, concatenate, cut every source_length tokens; no discarded per-document remainder",
                "dtype": "<u2"}
    final = out / "manifest.json"
    if final.exists():
        manifest = read_json(final)
        if manifest["identity"] != identity or manifest["status"] != "complete":
            raise ValueError("Prepared C4 identity mismatch")
        return manifest
    state_file = out / "preparation.json"
    if state_file.exists():
        state = read_json(state_file)
        if state["identity"] != identity:
            raise ValueError("Resume preparation identity mismatch")
    else:
        state = {"identity": identity, "tokens": 0, "documents": 0, "document_index_bytes": 0,
                 "completed_shards": [], "started_at": time.time()}
        atomic_json(state_file, state)
    prefix = "en/c4-train." if split == "train" else "en/c4-validation."
    shards = sorted(p for p in HfApi().list_repo_files(spec["id"], repo_type="dataset", revision=spec["revision"])
                    if p.startswith(prefix) and p.endswith(".json.gz"))
    if not shards:
        raise ValueError("No pinned English C4 source shards")
    wanted = examples * source_length
    tokens_file, docs_file = out / "tokens.u16", out / "documents.jsonl"
    for path, length in ((tokens_file, state["tokens"] * 2), (docs_file, state["document_index_bytes"])):
        if not path.exists():
            path.touch()
        with path.open("r+b") as f:
            if os.fstat(f.fileno()).st_size < length:
                raise ValueError("Committed preparation output is truncated")
            f.truncate(length)
    start = len(state["completed_shards"])
    with tokens_file.open("ab") as token_out, docs_file.open("ab") as docs_out:
        for shard in shards[start:]:
            if state["tokens"] == wanted:
                break
            source_path = hf_hub_download(spec["id"], shard, revision=spec["revision"], repo_type="dataset",
                                          cache_dir=str(root / "downloads"))
            before = state["tokens"]
            last_row = -1
            started = time.monotonic()
            with gzip.open(source_path, "rt") as f:
                batch = []
                for row, line in enumerate(f):
                    document = json.loads(line)
                    batch.append((row, document["text"]))
                    if len(batch) < batch_docs:
                        continue
                    last_row = write_documents(batch, tokenizer, token_out, docs_out, shard, state, wanted)
                    batch.clear()
                    if state["tokens"] == wanted:
                        break
                if batch and state["tokens"] < wanted:
                    last_row = write_documents(batch, tokenizer, token_out, docs_out, shard, state, wanted)
            token_out.flush()
            docs_out.flush()
            os.fsync(token_out.fileno())
            os.fsync(docs_out.fileno())
            state["document_index_bytes"] = docs_out.tell()
            state["completed_shards"].append({"path": shard, "sha256": file_hash(source_path),
                                               "first_token": before, "end_token": state["tokens"],
                                               "last_source_row": last_row,
                                               "seconds": time.monotonic() - started})
            atomic_json(state_file, state)
            print(json.dumps({"split": split, "source_shard": shard, "tokens": state["tokens"],
                              "required_tokens": wanted, "elapsed_seconds": time.time() - state["started_at"]}), flush=True)
    if state["tokens"] != wanted:
        raise ValueError("C4 source exhausted before required prefix")
    manifest = {"identity": identity, "status": "complete", "tokens": wanted, "examples": examples,
                "source_length": source_length, "token_sha256": file_hash(tokens_file),
                "documents_sha256": file_hash(docs_file), "source_shards": state["completed_shards"],
                "documents": state["documents"], "eos_token_id": tokenizer.eos_token_id,
                "special_ids": tokenizer.all_special_ids,
                "elapsed_seconds": time.time() - state["started_at"]}
    manifest["content_hash"] = object_hash(manifest)
    atomic_json(final, manifest)
    return manifest


def write_documents(batch, tokenizer, token_out, docs_out, shard, state, wanted):
    texts = [text for _, text in batch]
    encodings = tokenizer(texts, add_special_tokens=False, truncation=False, return_attention_mask=False)["input_ids"]
    last = -1
    for (row, text), ids in zip(batch, encodings):
        if state["tokens"] >= wanted:
            break
        ids = [*ids, tokenizer.eos_token_id]
        if min(ids) < 0 or max(ids) >= 65536:
            raise ValueError("Token out of uint16 range")
        keep = min(len(ids), wanted - state["tokens"])
        record = {"shard": shard, "row": row, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "start_token": state["tokens"], "end_token": state["tokens"] + keep,
                  "full_document_tokens_with_eos": len(ids)}
        token_out.write(np.asarray(ids[:keep], dtype="<u2").tobytes())
        docs_out.write((json.dumps(record, sort_keys=True) + "\n").encode())
        state["tokens"] += keep
        state["documents"] += 1
        last = row
    return last


def prepare_owt(root):
    root = Path(root)
    spec = sources()["owt"]
    path = root / "owt"
    marker = root / "owt_manifest.json"
    if marker.exists():
        if read_json(marker)["source"] != spec:
            raise ValueError("OWT revision mismatch")
        return path
    snapshot_download(spec["id"], revision=spec["revision"], repo_type="dataset", local_dir=str(path),
                      allow_patterns=["*.arrow", "*.json"], max_workers=4)
    from datasets import load_from_disk
    ds = load_from_disk(str(path))
    atomic_json(marker, directory_manifest(path) | {"source": spec, "examples": len(ds), "status": "complete",
                                                  "calibration_ids": list(range(4096)), "max_length": 1024})
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", choices=["assets", "c4", "owt", "all", "evaluator"], default="all")
    parser.add_argument("--examples", type=int, help="Small preparation fixture only; production defaults to fixed budget")
    parser.add_argument("--batch-docs", type=int, default=256)
    args = parser.parse_args()
    root = Path(args.root)
    with preparation_lock(root / f"prepare-{args.stage}.lock"):
        if args.stage == "evaluator":
            model_snapshot(root, "evaluator")
            return
        t5 = model_snapshot(root)
        if args.stage in ("c4", "all"):
            tokenizer = AutoTokenizer.from_pretrained(t5)
            cfg = load_config("src/configs/t5_adaptation/arm_c.yml")
            # Each split has its own lock even if callers use different stages.
            with preparation_lock(root / "c4.lock"):
                prepare_c4(root, "train", args.examples or cfg["source_examples"], 568, tokenizer, args.batch_docs)
                prepare_c4(root, "validation", 4096, 1023, tokenizer, args.batch_docs)
        if args.stage in ("owt", "all"):
            with preparation_lock(root / "owt.lock"):
                prepare_owt(root)


if __name__ == "__main__":
    main()

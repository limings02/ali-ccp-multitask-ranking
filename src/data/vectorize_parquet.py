from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.types as pat


@dataclass(frozen=True)
class FieldLayout:
    name: str
    idx_col: str
    val_col: Optional[str]
    is_multi_hot: bool
    use_value: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _open_memmap(path: Path, dtype: str | np.dtype, shape: Tuple[int, ...]) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(dtype), shape=shape)


def _iter_batches(
    dataset: ds.Dataset,
    columns: List[str],
    batch_rows: int,
    max_rows: Optional[int],
) -> Iterator[pa.RecordBatch]:
    scanner = dataset.scanner(columns=columns, batch_size=batch_rows)
    emitted = 0
    for batch in scanner.to_batches():
        if max_rows is not None:
            remain = max_rows - emitted
            if remain <= 0:
                break
            if batch.num_rows > remain:
                batch = batch.slice(0, remain)
        if batch.num_rows == 0:
            continue
        emitted += batch.num_rows
        yield batch


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse json: {path}") from exc


def _resolve_field_layouts(source_meta: Dict[str, Any], schema: pa.Schema) -> List[FieldLayout]:
    feature_meta = source_meta.get("feature_meta") or {}
    schema_names = set(schema.names)
    schema_map = {name: schema.field(name) for name in schema.names}

    if feature_meta:
        # Match current parquet collator ordering: it sorts by "*_idx" key names, then strips suffix.
        field_names = sorted(feature_meta.keys(), key=lambda base: f"{base}_idx")
    else:
        field_names = sorted(
            name[:-4]
            for name in schema.names
            if name.startswith("f") and name.endswith("_idx")
        )

    layouts: List[FieldLayout] = []
    for name in field_names:
        idx_col = f"{name}_idx"
        if idx_col not in schema_names:
            raise KeyError(f"Missing idx column in parquet schema: {idx_col}")

        val_col = f"{name}_val" if f"{name}_val" in schema_names else None
        meta = feature_meta.get(name, {})
        use_value = bool(meta.get("use_value", val_col is not None))
        is_multi = bool(meta.get("is_multi_hot", pat.is_list(schema_map[idx_col].type)))

        if use_value and val_col is None:
            raise KeyError(f"Field {name} expects value column but {name}_val is missing")

        layouts.append(
            FieldLayout(
                name=name,
                idx_col=idx_col,
                val_col=val_col,
                is_multi_hot=is_multi,
                use_value=use_value,
            )
        )
    return layouts


def _to_numpy(col: pa.Array | pa.ChunkedArray, dtype: np.dtype) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    arr = col.to_numpy(zero_copy_only=False)
    return np.asarray(arr, dtype=dtype)


def _extract_lengths(col: pa.Array | pa.ChunkedArray, is_multi_hot: bool, rows: int) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if is_multi_hot:
        lengths = pc.list_value_length(col)
        if lengths.null_count > 0:
            lengths = pc.fill_null(lengths, 0)
        out = lengths.to_numpy(zero_copy_only=False)
        return np.asarray(out, dtype=np.int64)
    if col.null_count > 0:
        null_mask = pc.is_null(col).to_numpy(zero_copy_only=False)
        return (~null_mask).astype(np.int64, copy=False)
    return np.ones(rows, dtype=np.int64)


def _flatten_int64(col: pa.Array | pa.ChunkedArray, is_multi_hot: bool) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if is_multi_hot:
        flat = pc.list_flatten(col)
        if flat.null_count > 0:
            out = np.asarray([int(x) for x in flat.to_pylist() if x is not None], dtype=np.int64)
            return out
        out = flat.to_numpy(zero_copy_only=False)
        return np.asarray(out, dtype=np.int64)

    if col.null_count > 0:
        vals = [int(x) for x in col.to_pylist() if x is not None]
        return np.asarray(vals, dtype=np.int64)
    out = col.to_numpy(zero_copy_only=False)
    return np.asarray(out, dtype=np.int64).reshape(-1)


def _flatten_float32(col: pa.Array | pa.ChunkedArray, is_multi_hot: bool) -> np.ndarray:
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    if is_multi_hot:
        flat = pc.list_flatten(col)
        if flat.null_count > 0:
            out = np.asarray([float(x) for x in flat.to_pylist() if x is not None], dtype=np.float32)
            return out
        out = flat.to_numpy(zero_copy_only=False)
        return np.asarray(out, dtype=np.float32)

    if col.null_count > 0:
        vals = [float(x) for x in col.to_pylist() if x is not None]
        return np.asarray(vals, dtype=np.float32)
    out = col.to_numpy(zero_copy_only=False)
    return np.asarray(out, dtype=np.float32).reshape(-1)


def _field_file_names(field_name: str) -> Dict[str, str]:
    return {
        "indices_file": f"f__{field_name}__indices.npy",
        "offsets_file": f"f__{field_name}__offsets.npy",
        "weights_file": f"f__{field_name}__weights.npy",
    }


def _write_split(
    split: str,
    processed_root: Path,
    out_root: Path,
    field_layouts: List[FieldLayout],
    source_meta: Dict[str, Any],
    data_version: str,
    batch_rows: int,
    max_rows: Optional[int],
) -> Dict[str, Any]:
    split_in = processed_root / split
    if not split_in.exists():
        raise FileNotFoundError(f"Processed split not found: {split_in}")

    split_out = out_root / split
    split_out.mkdir(parents=True, exist_ok=True)
    dataset = ds.dataset(split_in, format="parquet")

    expected_rows = int(source_meta.get("rows", {}).get(split, dataset.count_rows()))
    num_rows = min(expected_rows, max_rows) if max_rows is not None else expected_rows
    if num_rows <= 0:
        raise ValueError(f"Split {split} has no rows to vectorize")

    y_ctr_mm = _open_memmap(split_out / "y_ctr.npy", np.float32, (num_rows,))
    y_cvr_mm = _open_memmap(split_out / "y_cvr.npy", np.float32, (num_rows,))
    y_ctcvr_mm = _open_memmap(split_out / "y_ctcvr.npy", np.float32, (num_rows,))
    click_mask_mm = _open_memmap(split_out / "click_mask.npy", np.float32, (num_rows,))
    row_id_mm = _open_memmap(split_out / "row_id.npy", np.int64, (num_rows,))
    entity_offsets_mm = _open_memmap(split_out / "entity_id_offsets.npy", np.int64, (num_rows + 1,))
    entity_offsets_mm[0] = 0

    field_offsets_mm: Dict[str, np.memmap] = {}
    for layout in field_layouts:
        files = _field_file_names(layout.name)
        off = _open_memmap(split_out / files["offsets_file"], np.int64, (num_rows + 1,))
        off[0] = 0
        field_offsets_mm[layout.name] = off

    field_nnz: Dict[str, int] = {layout.name: 0 for layout in field_layouts}
    row_cursor = 0
    pass1_columns = [
        "y_ctr",
        "y_cvr",
        "y_ctcvr",
        "click_mask",
        "row_id",
        "entity_id",
        *[layout.idx_col for layout in field_layouts],
    ]

    entity_bin_path = split_out / "entity_id_data.bin"
    with entity_bin_path.open("wb") as entity_writer:
        # Pass1: only compute row-level offsets and labels/entity arrays.
        # We do not materialize flattened indices yet, so memory stays bounded by one record batch.
        for batch in _iter_batches(dataset, pass1_columns, batch_rows=batch_rows, max_rows=max_rows):
            rows = int(batch.num_rows)
            next_cursor = row_cursor + rows
            if next_cursor > num_rows:
                raise RuntimeError(f"Row overflow in pass1: {next_cursor} > {num_rows}")

            y_ctr_batch = _to_numpy(batch.column("y_ctr"), np.float32)
            y_cvr_batch = _to_numpy(batch.column("y_cvr"), np.float32)
            y_ctr_mm[row_cursor:next_cursor] = y_ctr_batch
            y_cvr_mm[row_cursor:next_cursor] = y_cvr_batch

            if "y_ctcvr" in batch.schema.names:
                y_ctcvr_mm[row_cursor:next_cursor] = _to_numpy(batch.column("y_ctcvr"), np.float32)
            else:
                y_ctcvr_mm[row_cursor:next_cursor] = np.logical_and(
                    y_ctr_batch > 0.5,
                    y_cvr_batch > 0.5,
                ).astype(np.float32, copy=False)

            click_mask_mm[row_cursor:next_cursor] = _to_numpy(batch.column("click_mask"), np.float32)
            row_id_mm[row_cursor:next_cursor] = _to_numpy(batch.column("row_id"), np.int64)

            entity_vals = batch.column("entity_id").to_pylist()
            serialized = [
                json.dumps(v, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                for v in entity_vals
            ]
            entity_blob = b"".join(serialized)
            entity_writer.write(entity_blob)
            lengths = np.fromiter((len(x) for x in serialized), dtype=np.int64, count=rows)
            entity_offsets_mm[row_cursor + 1: next_cursor + 1] = entity_offsets_mm[row_cursor] + np.cumsum(lengths)

            for layout in field_layouts:
                idx_col = batch.column(layout.idx_col)
                lengths_arr = _extract_lengths(idx_col, is_multi_hot=layout.is_multi_hot, rows=rows)
                offsets = field_offsets_mm[layout.name]
                base = field_nnz[layout.name]
                offsets[row_cursor + 1: next_cursor + 1] = base + np.cumsum(lengths_arr, dtype=np.int64)
                field_nnz[layout.name] = int(offsets[next_cursor])

            row_cursor = next_cursor

    if row_cursor != num_rows:
        raise RuntimeError(f"Pass1 row count mismatch: got {row_cursor}, expected {num_rows}")

    y_ctr_mm.flush()
    y_cvr_mm.flush()
    y_ctcvr_mm.flush()
    click_mask_mm.flush()
    row_id_mm.flush()
    entity_offsets_mm.flush()
    for mm in field_offsets_mm.values():
        mm.flush()

    field_indices_mm: Dict[str, np.memmap] = {}
    field_weights_mm: Dict[str, Optional[np.memmap]] = {}
    for layout in field_layouts:
        files = _field_file_names(layout.name)
        nnz_total = int(field_nnz[layout.name])
        field_indices_mm[layout.name] = _open_memmap(
            split_out / files["indices_file"],
            np.int64,
            (nnz_total,),
        )
        if layout.use_value:
            field_weights_mm[layout.name] = _open_memmap(
                split_out / files["weights_file"],
                np.float32,
                (nnz_total,),
            )
        else:
            field_weights_mm[layout.name] = None

    pass2_columns = [layout.idx_col for layout in field_layouts]
    pass2_columns.extend(layout.val_col for layout in field_layouts if layout.use_value and layout.val_col)
    field_cursor = {layout.name: 0 for layout in field_layouts}

    # Pass2: flatten each field and fill pre-allocated memmaps directly.
    for batch in _iter_batches(dataset, pass2_columns, batch_rows=batch_rows, max_rows=max_rows):
        for layout in field_layouts:
            idx_flat = _flatten_int64(batch.column(layout.idx_col), is_multi_hot=layout.is_multi_hot)
            nnz = int(idx_flat.shape[0])
            cur = field_cursor[layout.name]
            nxt = cur + nnz
            cap = int(field_indices_mm[layout.name].shape[0])
            if nxt > cap:
                raise ValueError(
                    f"{layout.name}: pass2 nnz overflow cur={cur} nnz_batch={nnz} cap={cap} "
                    f"is_multi_hot={layout.is_multi_hot}"
                )
            field_indices_mm[layout.name][cur:nxt] = idx_flat

            w_mm = field_weights_mm[layout.name]
            if w_mm is not None:
                if layout.val_col is None:
                    val_flat = np.ones(nnz, dtype=np.float32)
                else:
                    val_flat = _flatten_float32(batch.column(layout.val_col), is_multi_hot=layout.is_multi_hot)
                    if val_flat.shape[0] != nnz:
                        if val_flat.shape[0] == 0 and nnz > 0:
                            val_flat = np.ones(nnz, dtype=np.float32)
                        else:
                            raise ValueError(
                                f"{layout.name}: value/idx nnz mismatch in pass2 ({val_flat.shape[0]} vs {nnz})"
                            )
                w_mm[cur:nxt] = val_flat

            field_cursor[layout.name] = nxt

    for layout in field_layouts:
        nnz_total = int(field_nnz[layout.name])
        if field_cursor[layout.name] != nnz_total:
            raise RuntimeError(
                f"{layout.name}: pass2 nnz mismatch ({field_cursor[layout.name]} vs {nnz_total})"
            )
        field_indices_mm[layout.name].flush()
        if field_weights_mm[layout.name] is not None:
            field_weights_mm[layout.name].flush()

    field_meta: Dict[str, Any] = {}
    for layout in field_layouts:
        files = _field_file_names(layout.name)
        field_meta[layout.name] = {
            "is_multi_hot": bool(layout.is_multi_hot),
            "use_value": bool(layout.use_value),
            "nnz_total": int(field_nnz[layout.name]),
            "indices_file": files["indices_file"],
            "offsets_file": files["offsets_file"],
            "weights_file": files["weights_file"] if layout.use_value else None,
            "indices_dtype": "int64",
            "offsets_dtype": "int64",
            "weights_dtype": "float32" if layout.use_value else None,
        }

    split_meta = {
        "format": "vectorized_ragged_csr_v1",
        "split": split,
        "data_version": data_version,
        "vocab_version": str(source_meta.get("featuremap_hash", "unknown")),
        "hash_version": str(source_meta.get("featuremap_hash", "unknown")),
        "num_rows": int(num_rows),
        "field_names": [layout.name for layout in field_layouts],
        "offsets_definition": {
            "stored": "[N+1]",
            "training_collate": "[B] for include_last_offset=False",
            "note": "offsets[i] is bag start; offsets[N] equals total nnz",
        },
        "labels": {
            "y_ctr": {"file": "y_ctr.npy", "shape": [int(num_rows)], "dtype": "float32"},
            "y_cvr": {"file": "y_cvr.npy", "shape": [int(num_rows)], "dtype": "float32"},
            "y_ctcvr": {"file": "y_ctcvr.npy", "shape": [int(num_rows)], "dtype": "float32"},
            "click_mask": {"file": "click_mask.npy", "shape": [int(num_rows)], "dtype": "float32"},
            "row_id": {"file": "row_id.npy", "shape": [int(num_rows)], "dtype": "int64"},
            "entity_id": {
                "data_file": "entity_id_data.bin",
                "offsets_file": "entity_id_offsets.npy",
                "offsets_shape": [int(num_rows) + 1],
                "encoding": "json-utf8-bytes",
            },
        },
        "fields": field_meta,
        "generated_at": _utc_now(),
    }

    (split_out / "metadata.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return split_meta


def vectorize_processed(
    processed_root: Path,
    source_metadata_path: Path,
    out_root: Path,
    splits: List[str],
    batch_rows: int,
    max_rows: Optional[int] = None,
    overwrite: bool = False,
    data_version: str = "vectorized_v1",
) -> Dict[str, Any]:
    if not processed_root.exists():
        raise FileNotFoundError(f"Processed root not found: {processed_root}")
    if not source_metadata_path.exists():
        raise FileNotFoundError(f"Source metadata not found: {source_metadata_path}")

    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_meta = _load_json(source_metadata_path)
    sample_split = splits[0]
    schema = ds.dataset(processed_root / sample_split, format="parquet").schema
    field_layouts = _resolve_field_layouts(source_meta, schema=schema)

    split_summaries: Dict[str, Any] = {}
    for split in splits:
        split_summaries[split] = _write_split(
            split=split,
            processed_root=processed_root,
            out_root=out_root,
            field_layouts=field_layouts,
            source_meta=source_meta,
            data_version=data_version,
            batch_rows=batch_rows,
            max_rows=max_rows,
        )

    # Keep model feature metadata compatible with existing build_model_feature_meta().
    root_meta = dict(source_meta)
    root_meta.update(
        {
            "format": "vectorized_ragged_csr_v1",
            "generated_at": _utc_now(),
            "data_version": data_version,
            "vocab_version": str(source_meta.get("featuremap_hash", "unknown")),
            "hash_version": str(source_meta.get("featuremap_hash", "unknown")),
            "vectorized": {
                "root": str(out_root),
                "splits": {
                    split: {
                        "num_rows": int(summary["num_rows"]),
                        "path": str(out_root / split),
                        "metadata": str(out_root / split / "metadata.json"),
                    }
                    for split, summary in split_summaries.items()
                },
                "field_names": split_summaries[sample_split]["field_names"],
                "offsets_definition": split_summaries[sample_split]["offsets_definition"],
            },
        }
    )

    (out_root / "metadata.json").write_text(
        json.dumps(root_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return root_meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert processed parquet to vectorized CSR-style training data."
    )
    parser.add_argument("--processed-root", type=str, default="data/processed")
    parser.add_argument("--source-metadata", type=str, default="data/processed/metadata.json")
    parser.add_argument("--out-root", type=str, default="data/vectorized")
    parser.add_argument("--splits", nargs="+", default=["train", "valid"])
    parser.add_argument("--batch-rows", type=int, default=65536)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-version", type=str, default="vectorized_v1")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root_meta = vectorize_processed(
        processed_root=Path(args.processed_root),
        source_metadata_path=Path(args.source_metadata),
        out_root=Path(args.out_root),
        splits=[str(s) for s in args.splits],
        batch_rows=int(args.batch_rows),
        max_rows=(int(args.max_rows) if args.max_rows is not None else None),
        overwrite=bool(args.overwrite),
        data_version=str(args.data_version),
    )
    print(
        json.dumps(
            {
                "format": root_meta.get("format"),
                "data_version": root_meta.get("data_version"),
                "vectorized_root": str(Path(args.out_root)),
                "splits": root_meta.get("vectorized", {}).get("splits", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Stage 1: stream the FAIR Universe HiggsML public dataset, keep a configurable fraction of each process, apply the configured systematic variations chunk by chunk, and write one ROOT file per variation with one tree per process.

The public parquet (220M rows in 210 row groups) is never loaded whole. Row groups are read in runs of ``data.chunk_rows`` rows; each chunk is filtered to the requested processes and the per-process subsample, passed through HiggsML ``systematics()`` once per variation (nominal plus every entry of ``systematics``), cut with ``preprocess.selections`` and appended to the output trees. Peak memory is a few copies of one chunk, independent of the dataset size.

Per-process subsampling: an event of process ``p`` is kept iff ``u < fractions[p]``, where ``u`` is drawn from one seeded stream consumed row group by row group. Kept weights are rescaled per process by (sum of weights of every event of ``p`` in the training region) / (sum over the kept events), computed in a weights-only pass, so each process retains the yield of the training region. With ``preprocess.balance: true`` the fractions are additionally reduced so every process keeps the same expected number of events (the former balance-to-smallest behaviour, now exact over the training region).
"""

import argparse
import gc
import logging
import os
import resource
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import uproot
from HiggsML.datasets import download_dataset
from HiggsML.systematics import systematics

from nsbi_common_utils.configuration import load_pipeline_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    return load_pipeline_config(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream the HiggsML dataset, subsample per process, apply systematics and write ROOT files.")
    parser.add_argument("--config", type=str, default="config.pipeline.yaml", help="Configuration file path.")
    return parser.parse_args()


def log_rss(context: str) -> None:
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    logger.info(f"[memory] peak RSS after {context}: {peak_gb:.1f} GB")


def open_source(url: str):
    """Return the ParquetFile handle and the metadata dict of the public dataset. ``download_dataset`` skips the download when ``./data.zip`` or an extracted ``./data`` already exists in the working directory (the CHTC job wrapper symlinks the staged zip there)."""
    logger.info(f"Opening dataset from {url}")
    data = download_dataset(url)
    return data.parquet_file, dict(getattr(data, "metadata", {}) or {})


def resolve_fractions(data_cfg: dict, processes: list) -> dict:
    """Per-process keep fractions from ``data.fractions``; a process missing from the mapping is kept in full."""
    fractions = data_cfg.get("fractions") or {}
    out = {}
    for p in processes:
        f = float(fractions.get(p, 1.0))
        if not 0.0 < f <= 1.0:
            raise ValueError(f"data.fractions[{p}] must be in (0, 1], got {f}")
        out[p] = f
    return out


def select_events(pf: pq.ParquetFile, fractions: dict, seed: int, n_test: int = 0) -> tuple:
    """Weights-only pass over the whole file.

    Draws one uniform number per row from a stream seeded with ``seed`` and consumed in row-group order, keeps rows of process ``p`` with ``u < fractions[p]``, and accumulates per-process counts and weight sums before and after the selection. Rows with global index below ``n_test`` belong to the challenge's reserved test region: they are never kept, and they are tallied separately under ``n_test``/``w_test`` instead of ``n_all``/``w_all``. ``w_region`` is the total weight of every label (including processes that are not requested) in the training region, needed for the global rescale. Returns the per-process statistics, the training-region total and the boolean keep mask of every row group.
    """
    rng = np.random.default_rng(seed)
    masks = {}
    stats = {p: {"n_all": 0, "w_all": 0.0, "n_kept": 0, "w_kept": 0.0, "n_test": 0, "w_test": 0.0} for p in fractions}
    w_region = 0.0
    offset = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["detailed_labels", "weights"])
        labels = tbl.column("detailed_labels")
        w = tbl.column("weights").to_numpy().astype(np.float64)
        u = rng.random(len(w))
        in_train = (np.arange(offset, offset + len(w)) >= n_test)
        offset += len(w)
        w_region += float(w[in_train].sum())
        keep = np.zeros(len(w), dtype=bool)
        for p, f in fractions.items():
            is_p = np.asarray(pc.equal(labels, p).to_numpy(zero_copy_only=False), dtype=bool)
            is_p_train = is_p & in_train
            kept_p = is_p_train & (u < f)
            keep |= kept_p
            stats[p]["n_all"] += int(is_p_train.sum())
            stats[p]["w_all"] += float(w[is_p_train].sum())
            stats[p]["n_kept"] += int(kept_p.sum())
            stats[p]["w_kept"] += float(w[kept_p].sum())
            stats[p]["n_test"] += int((is_p & ~in_train).sum())
            stats[p]["w_test"] += float(w[is_p & ~in_train].sum())
        masks[rg] = keep
    return stats, w_region, masks


def iter_chunks(pf: pq.ParquetFile, chunk_rows: int, masks: dict):
    """Yield (source rows read, pandas frame of kept rows) for consecutive runs of whole row groups of at least ``chunk_rows`` source rows (except the last). Row groups whose mask keeps nothing (in particular those lying entirely inside the reserved test region) are counted but not read. Every kept row carries its global row index in the source file as ``event_id``, which is what identifies the same event across the nominal and systematic-variation trees (k-fold assignment keys on it)."""
    tables, n, offset = [], 0, 0
    for rg in range(pf.num_row_groups):
        n_rg = len(masks[rg])
        if not masks[rg].any():
            n += n_rg
            offset += n_rg
            continue
        tbl = pf.read_row_group(rg)
        n += tbl.num_rows
        tbl = tbl.append_column("event_id", pa.array(np.arange(offset, offset + n_rg, dtype=np.int64)))
        offset += n_rg
        tables.append(tbl.filter(pa.array(masks[rg])))
        if n >= chunk_rows:
            yield n, pa.concat_tables(tables).to_pandas()
            tables, n = [], 0
    if tables:
        yield n, pa.concat_tables(tables).to_pandas()
    elif n:
        yield n, pd.DataFrame()


class RootWriter:
    """One ROOT file per variation; each process tree is created on first write as a classic TTree and extended chunk by chunk. The file is written under a temporary name and moved into place on close. TTrees are used explicitly because uproot >= 5.7 writes RNTuples by default and mis-sizes the key of an RNTuple created past the 2 GB mark (scikit-hep/uproot5#1828, unreleased as of 5.7.7)."""

    def __init__(self, path: str):
        self.path = path
        self.tmp_path = path + ".tmp"
        self.file = uproot.recreate(self.tmp_path)
        self.trees = set()
        self.stats = {}

    def write(self, tree_name: str, df: pd.DataFrame) -> None:
        arrays = {col: df[col].to_numpy() for col in df.columns}
        if tree_name not in self.trees:
            self.file.mktree(tree_name, {col: arr.dtype for col, arr in arrays.items()})
            self.trees.add(tree_name)
        self.file[tree_name].extend(arrays)
        n, s = self.stats.get(tree_name, (0, 0.0))
        self.stats[tree_name] = (n + len(df), s + float(df["weights"].sum()))

    def close(self) -> None:
        self.file.close()
        os.replace(self.tmp_path, self.path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)["data_loader"]
    data_cfg = config["data"]
    pre_cfg = config["preprocess"]

    processes = list(data_cfg["processes"])
    seed = int(pre_cfg["seed"])
    balance = bool(pre_cfg.get("balance", False))
    selections = pre_cfg["selections"]
    chunk_rows = int(data_cfg.get("chunk_rows", 2_000_000))
    output_dir = config["output"]["dir"]

    # Inject the shared seed into each systematic's settings, as before; TES/JES are deterministic, the seed only matters for soft-MET smearing.
    syst_settings = {name: dict(syst_args) for name, syst_args in config["systematics"].items()}
    for syst_args in syst_settings.values():
        syst_args["seed"] = seed
    variations = {"nominal": {}}
    variations.update(syst_settings)

    pf, metadata = open_source(data_cfg["url"])
    total_rows = int(metadata.get("total_rows", pf.metadata.num_rows))
    logger.info(f"Source: {total_rows} rows in {pf.num_row_groups} row groups; {chunk_rows} rows per chunk")

    # Same split as HiggsML.datasets.Data(test_size=...): the first int(test_size * total_rows) rows are the challenge's held-out test region and are never used for training.
    test_size = float(data_cfg.get("test_size", 0.3))
    if not 0.0 <= test_size < 1.0:
        raise ValueError(f"data.test_size must be in [0, 1), got {test_size}")
    n_test = int(test_size * total_rows)
    logger.info(f"Reserved test region: first {n_test} rows (test_size={test_size}); training region: {total_rows - n_test} rows")

    fractions = resolve_fractions(data_cfg, processes)
    logger.info(f"Per-process keep fractions: {fractions}")
    stats, w_region, masks = select_events(pf, fractions, seed, n_test)

    if balance:
        # Reduce every fraction so all processes keep the same expected count as the smallest one. Re-running the pass with the same seed keeps the selection nested inside the first one.
        target = min(stats[p]["n_kept"] for p in processes)
        fractions = {p: fractions[p] * target / stats[p]["n_kept"] for p in processes}
        logger.info(f"Balancing enabled: every process reduced to ~{target} events; effective fractions {fractions}")
        stats, w_region, masks = select_events(pf, fractions, seed, n_test)

    # HiggsML.datasets.Data.load_train_set rescales whatever it loads by one global factor so that the total weight equals the metadata sum_weights; reproduce that so the yields match the challenge's convention regardless of test_size.
    sum_weights = float(metadata.get("sum_weights", 0.0))
    global_factor = sum_weights / w_region if sum_weights > 0.0 and w_region > 0.0 else 1.0
    logger.info(f"Training region holds {w_region:.1f} of the metadata sum_weights {sum_weights:.1f}: global weight factor {global_factor:.6f}")

    factor = {}
    logger.info("Per-process selection (training region only):")
    for p in processes:
        s = stats[p]
        if s["w_kept"] <= 0.0:
            raise ValueError(f"No events of process '{p}' selected; check data.fractions")
        factor[p] = global_factor * s["w_all"] / s["w_kept"]
        logger.info(f"  {p:10s} all={s['n_all']:>11d}  kept={s['n_kept']:>10d} ({100.0 * s['n_kept'] / max(s['n_all'], 1):6.2f}%)  "
                    f"region yield={s['w_all']:12.1f} -> {global_factor * s['w_all']:12.1f}  weight factor={factor[p]:.6f}  "
                    f"[test region: {s['n_test']} events, weight {s['w_test']:.1f}]")
    log_rss("selection pass")

    os.makedirs(output_dir, exist_ok=True)
    writers = {name: RootWriter(os.path.join(output_dir, f"dataset_{name}.root")) for name in variations}

    n_seen = 0
    try:
        for ichunk, (n_read, df_chunk) in enumerate(iter_chunks(pf, chunk_rows, masks)):
            n_seen += n_read
            if df_chunk.empty:
                logger.info(f"Chunk {ichunk}: no selected events ({n_seen}/{total_rows} rows read)")
                continue
            df_chunk["weights"] = df_chunk["weights"].to_numpy() * df_chunk["detailed_labels"].map(factor).to_numpy()
            logger.info(f"Chunk {ichunk}: {len(df_chunk)} selected events ({n_seen}/{total_rows} rows read)")

            for name, syst_args in variations.items():
                df_var = systematics(data_set=df_chunk, dopostprocess=False, **syst_args)
                df_var = df_var.query(selections)
                for p in processes:
                    df_p = df_var[df_var["detailed_labels"] == p]
                    if len(df_p) == 0:
                        continue
                    columns_to_keep = [c for c in df_p.columns if c != "detailed_labels"]
                    writers[name].write(f"tree_{p}", df_p[columns_to_keep])
                del df_var
                gc.collect()

            del df_chunk
            gc.collect()
            log_rss(f"chunk {ichunk}")
    finally:
        for writer in writers.values():
            writer.close()

    if n_seen != total_rows:
        raise RuntimeError(f"Read {n_seen} rows but the source has {total_rows}")

    for name, writer in writers.items():
        for tree_name, (n, s) in sorted(writer.stats.items()):
            logger.info(f"  {name:10s} {tree_name:14s} events={n:>10d}  sum of weights={s:12.1f}")
        logger.info(f"Wrote {writer.path}")

    # Never leave the extracted parquet behind: HTCondor would ship it back to the submit node.
    for leftover in ("data", "data.zip"):
        if os.path.islink(leftover) or os.path.isfile(leftover):
            os.remove(leftover)
        elif os.path.isdir(leftover):
            shutil.rmtree(leftover, ignore_errors=True)

    logger.info("Data loading workflow completed successfully.")


if __name__ == "__main__":
    main()

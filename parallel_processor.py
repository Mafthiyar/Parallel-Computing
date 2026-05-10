"""
parallel_processor.py
----------------------
Splits large DataFrames into chunks and processes them
in parallel using Python's multiprocessing module.
Demonstrates distributed/concurrent system design.
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def _process_chunk(args):
    """
    Worker function executed in each subprocess.
    Applies a transformation function to a DataFrame chunk.
    """
    chunk, transform_fn, chunk_id = args
    try:
        result = transform_fn(chunk)
        logger.debug(f"Chunk {chunk_id} processed: {len(result)} rows")
        return result
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id}: {e}")
        raise


class ParallelProcessor:
    """
    Splits a DataFrame into N chunks and processes each
    chunk in a separate process using a pool of workers.
    """

    def __init__(self, n_workers: int = None):
        """
        Args:
            n_workers: Number of parallel worker processes.
                       Defaults to number of CPU cores.
        """
        self.n_workers = n_workers or mp.cpu_count()
        logger.info(f"ParallelProcessor initialized with {self.n_workers} workers.")

    def process(self, df: pd.DataFrame, transform_fn: Callable) -> pd.DataFrame:
        """
        Process a DataFrame in parallel by splitting into chunks.

        Args:
            df: Input DataFrame to process.
            transform_fn: Function applied to each chunk independently.

        Returns:
            Concatenated result DataFrame.
        """
        chunks = self._split_into_chunks(df)
        args = [(chunk, transform_fn, i) for i, chunk in enumerate(chunks)]

        logger.info(f"Starting parallel processing: {len(chunks)} chunks, {self.n_workers} workers...")
        start = time.time()

        with mp.Pool(processes=self.n_workers) as pool:
            results = pool.map(_process_chunk, args)

        elapsed = time.time() - start
        result_df = pd.concat(results, ignore_index=True)
        logger.info(f"Parallel processing complete in {elapsed:.2f}s. Output: {result_df.shape}")
        return result_df

    def benchmark(self, df: pd.DataFrame, transform_fn: Callable) -> dict:
        """
        Compare parallel vs sequential processing time.

        Returns:
            Dict with timing results and speedup ratio.
        """
        # Sequential
        logger.info("Running sequential benchmark...")
        t0 = time.time()
        _ = transform_fn(df)
        sequential_time = time.time() - t0

        # Parallel
        logger.info("Running parallel benchmark...")
        t1 = time.time()
        _ = self.process(df, transform_fn)
        parallel_time = time.time() - t1

        speedup = sequential_time / parallel_time if parallel_time > 0 else 0

        result = {
            "rows": len(df),
            "workers": self.n_workers,
            "sequential_time_s": round(sequential_time, 4),
            "parallel_time_s": round(parallel_time, 4),
            "speedup": round(speedup, 2)
        }

        logger.info(f"Benchmark: Sequential={sequential_time:.2f}s | Parallel={parallel_time:.2f}s | Speedup={speedup:.2f}x")
        return result

    def _split_into_chunks(self, df: pd.DataFrame) -> list:
        """Split DataFrame into equal-sized chunks, one per worker."""
        chunk_size = max(1, len(df) // self.n_workers)
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        logger.info(f"Split {len(df)} rows into {len(chunks)} chunks (~{chunk_size} rows each)")
        return chunks


# Example transform function (can be replaced with any logic)
def clean_and_engineer_features(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Sample transformation applied per chunk:
    - Drop rows with all NaN
    - Add derived features
    """
    chunk = chunk.dropna(how="all")

    numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()

    # Feature: row-wise mean of numeric columns
    if numeric_cols:
        chunk["feature_mean"] = chunk[numeric_cols].mean(axis=1)
        chunk["feature_std"] = chunk[numeric_cols].std(axis=1)
        chunk["feature_range"] = chunk[numeric_cols].max(axis=1) - chunk[numeric_cols].min(axis=1)

    return chunk

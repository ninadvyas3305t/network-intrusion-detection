"""Dataset Acquisition Script for UNSW-NB15
Downloads the official research partitioned benchmark files:
- UNSW_NB15_training-set.csv
- UNSW_NB15_testing-set.csv
"""

import os
import sys
import requests
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

URL_SOURCES = [
    {
        "name": "Hugging Face (Mouwiya/UNSW-NB15-small)",
        "train": "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_training-set.csv",
        "test": "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_testing-set.csv",
    },
    {
        "name": "Hugging Face (Mouwiya/UNSW-NB15)",
        "train": "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
        "test": "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_testing-set.csv",
    },
]

def download_file(url: str, dest_path: str) -> bool:
    """Download a file with streaming and progress reporting using requests."""
    logger.info(f"Downloading from {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024 * 64):
                size = f.write(data)
                bar.update(size)

        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        logger.info(f"Successfully downloaded {dest_path} ({file_size_mb:.2f} MB)")
        return True
    except Exception as e:
        logger.warning(f"Failed to download from {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def acquire_unsw_nb15(data_raw_dir: str = "data/raw") -> bool:
    """Download training and testing CSVs into data_raw_dir."""
    os.makedirs(data_raw_dir, exist_ok=True)
    train_dest = os.path.join(data_raw_dir, "UNSW_NB15_training-set.csv")
    test_dest = os.path.join(data_raw_dir, "UNSW_NB15_testing-set.csv")

    train_ok = False
    test_ok = False

    # Check existing valid files
    if os.path.exists(train_dest) and os.path.getsize(train_dest) > 1000000:
        logger.info(f"Training set already present ({os.path.getsize(train_dest)/(1024*1024):.2f} MB). Skipping train download.")
        train_ok = True

    if os.path.exists(test_dest) and os.path.getsize(test_dest) > 1000000:
        logger.info(f"Testing set already present ({os.path.getsize(test_dest)/(1024*1024):.2f} MB). Skipping test download.")
        test_ok = True

    if train_ok and test_ok:
        logger.info("Both dataset splits already present.")
        return True

    for source in URL_SOURCES:
        logger.info(f"Attempting download from source: {source['name']}")
        if not train_ok:
            train_ok = download_file(source["train"], train_dest)
        if not test_ok:
            test_ok = download_file(source["test"], test_dest)

        if train_ok and test_ok:
            logger.info("Both dataset splits downloaded successfully.")
            return True
        else:
            logger.warning(f"Download from {source['name']} incomplete. Trying next source...")

    logger.error("Failed to acquire dataset files.")
    return False

if __name__ == "__main__":
    success = acquire_unsw_nb15()
    if not success:
        sys.exit(1)

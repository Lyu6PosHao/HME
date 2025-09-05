# src/hme/download.py
import argparse
import sys
from pathlib import Path
from huggingface_hub import snapshot_download


def download_model(repo_id: str, local_dir: Path):
    """Downloads a model from Hugging Face Hub to a local directory."""
    print(f"Downloading '{repo_id}' to '{local_dir}'...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            # It's good practice to ignore boilerplate files
            ignore_patterns=["*.md", ".gitattributes"],
        )
        print(f"Successfully downloaded '{repo_id}'.")
    except Exception as e:
        print(f"Failed to download '{repo_id}'. Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="A simple utility to download a model from Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="The repository ID on Hugging Face Hub (e.g., 'meta-llama/Meta-Llama-3-8B').",
    )
    parser.add_argument(
        "--local_dir",
        type=Path,
        required=True,
        help="The local directory to save the model to.",
    )
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    download_model(args.repo_id, args.local_dir)


if __name__ == "__main__":
    main()

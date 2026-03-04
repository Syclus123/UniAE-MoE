import argparse
from huggingface_hub import snapshot_download
from xares_llm.audiowebdataset import CACHE_DIR

def main():
    parser = argparse.ArgumentParser(
        description="Download and cache webdatasets specified in a XaresLLM training config."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=16, 
        help=f"Number of worker processes for the WebLoader. (Default: 16)"
    )
    args = parser.parse_args()
    snapshot_download(
        repo_id='mispeech/MECAT-Caption',
        local_dir=CACHE_DIR,
        repo_type='dataset',
        max_workers=args.num_workers
    )
    snapshot_download(
        repo_id='mispeech/xares_llm_data',
        local_dir=CACHE_DIR,
        repo_type='dataset',
        max_workers=args.num_workers
    )

if __name__ == "__main__":
    main()


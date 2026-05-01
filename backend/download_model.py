"""
Download embedding model to project directory using Chinese mirror.
"""
import os
from pathlib import Path

# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = Path("./models/all-MiniLM-L6-v2")


def download_model():
    """Download model to local project directory using Chinese mirror."""
    print(f"Downloading {MODEL_NAME}...")
    print(f"Target directory: {MODEL_DIR.absolute()}")

    # Create models directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Try multiple mirrors
    mirrors = [
        "https://hf-mirror.com",
        "https://mirrors.tuna.tsinghua.edu.cn/hugging-face",
    ]

    for mirror in mirrors:
        print(f"\nTrying mirror: {mirror}")
        os.environ["HF_ENDPOINT"] = mirror

        try:
            from huggingface_hub import snapshot_download

            # Download all model files to target directory
            downloaded_path = snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False
            )

            print(f"\nModel downloaded successfully!")
            print(f"Location: {downloaded_path}")

            # Verify model works
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(str(MODEL_DIR))
            test_embedding = model.encode("This is a test sentence")
            print(f"Embedding dimension: {len(test_embedding)}")
            print("Model verification passed!")

            return True

        except Exception as e:
            print(f"Failed with {mirror}: {str(e)[:200]}")
            continue

    print("\nAll mirrors failed.")
    return False


if __name__ == "__main__":
    success = download_model()
    if not success:
        print("\nPlease check your internet connection or try again later.")
        exit(1)

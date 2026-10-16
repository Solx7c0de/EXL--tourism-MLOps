import os
from huggingface_hub import HfApi, create_repo

DATA_PATH = "data/tourism.csv"
REPO_ID = os.environ["HF_DATASET_REPO"]
HF_TOKEN = os.environ["HF_TOKEN"]

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

api = HfApi(token=HF_TOKEN)

create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    private=True,
    exist_ok=True
)

api.upload_file(
    path_or_fileobj=DATA_PATH,
    path_in_repo="tourism.csv",
    repo_id=REPO_ID,
    repo_type="dataset"
)

print(f"Dataset successfully registered: {REPO_ID}")

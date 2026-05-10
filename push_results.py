import json
from pathlib import Path
from huggingface_hub import HfFileSystem, HfApi
import config

def push_daily_result(local_json_path: Path):
    fs = HfFileSystem(token=config.HF_TOKEN)
    remote_path = f"datasets/{config.OUTPUT_REPO}/dissipation_{config.TODAY}.json"
    try:
        with fs.open(remote_path, "w") as f:
            with open(local_json_path, "r") as local_f:
                f.write(local_f.read())
        print(f"✅ Results pushed to {remote_path}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        api = HfApi()
        try:
            api.create_repo(repo_id=config.OUTPUT_REPO, repo_type="dataset", exist_ok=True, token=config.HF_TOKEN)
            with fs.open(remote_path, "w") as f:
                with open(local_json_path, "r") as local_f:
                    f.write(local_f.read())
            print(f"✅ Repo created and results pushed")
        except Exception as e2:
            print(f"❌ Could not create repo or upload: {e2}")

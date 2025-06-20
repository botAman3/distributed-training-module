import os
import shutil
import time
import subprocess
from client.train.download_data import download_and_save_images

def download_from_db(target_dir="downloaded_code"):
    print("📥 Simulating download from DB...")
    download_and_save_images()
    
def run_client():
    print("🚀 Running federated client...")
    subprocess.run(["python", "downloaded_code/client.py"])

if __name__ == "__main__":
    download_from_db()
    time.sleep(3)  
    run_client()

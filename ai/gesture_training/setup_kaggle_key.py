"""
Quick helper — paste your Kaggle API key here and run this script
to automatically place kaggle.json in the right location.

HOW TO GET YOUR KEY:
  1. Go to https://www.kaggle.com/settings
  2. Scroll to 'API' section
  3. Click 'Create New Token'
  4. Open the downloaded kaggle.json with Notepad
  5. Copy username and key below
"""

KAGGLE_USERNAME = ""   # <-- paste your Kaggle username here
KAGGLE_KEY      = ""   # <-- paste your Kaggle API key here

# ── Auto-install ──────────────────────────────────────────────
import os, json

if not KAGGLE_USERNAME or not KAGGLE_KEY:
    print("\nERROR: Fill in KAGGLE_USERNAME and KAGGLE_KEY at the top of this script.")
    print("Get them from: https://www.kaggle.com/settings -> API -> Create New Token")
    exit(1)

kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
with open(kaggle_json_path, "w") as f:
    json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)

# Secure the file (Kaggle requires 600 permissions on Linux, ignored on Windows)
try:
    import stat
    os.chmod(kaggle_json_path, stat.S_IRUSR | stat.S_IWUSR)
except Exception:
    pass

print(f"\n[OK] kaggle.json saved to: {kaggle_json_path}")
print("[OK] You can now run: python download_and_train_asl.py")

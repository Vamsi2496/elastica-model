import os
import json
from .config import CONFIG_PATH

def save_config(python26_path, auto_dir_path):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump({
            "PYTHON26": python26_path,
            "AUTO_DIR": auto_dir_path
        }, f, indent=2)
    print(f"✓ Config saved to {CONFIG_PATH}")

def show_config():
    from .config import PYTHON26, AUTO_DIR
    print(f"\nCurrent config:")
    print(f"  PYTHON26 : {PYTHON26}")
    print(f"  AUTO_DIR : {AUTO_DIR}")
    print(f"  File     : {CONFIG_PATH}\n")

def main():
    print("\n=== Elastica Model — First Time Setup ===\n")
    print("Press Enter to keep the value shown in brackets.\n")

    from .config import PYTHON26, AUTO_DIR

    p26 = input(f"Path to Python 2.6 exe [{PYTHON26}]: ").strip()
    if not p26:
        p26 = PYTHON26

    adir = input(f"Path to AUTO 07p python dir [{AUTO_DIR}]: ").strip()
    if not adir:
        adir = AUTO_DIR

    if not os.path.exists(p26):
        print(f"  ⚠ Warning: '{p26}' not found — check path")
    if not os.path.exists(adir):
        print(f"  ⚠ Warning: '{adir}' not found — check path")

    save_config(p26, adir)
    print("\n✓ Setup complete. Run 'elastica-generate' to start.\n")

if __name__ == "__main__":
    main()      
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yml"

def load_config():
  with open(CONFIG_PATH,"r") as f:
    return yaml.safe_load(f)
  
   
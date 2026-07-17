import json
import sys
from pathlib import Path

# Add apps to sys.path so we can import faulttrace_api
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir / "apps" / "api"))

from faulttrace_api.main import app

def main():
    openapi_schema = app.openapi()
    docs_dir = root_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    out_path = docs_dir / "openapi.json"
    
    with open(out_path, "w") as f:
        json.dump(openapi_schema, f, indent=2)
    
    print(f"Exported OpenAPI schema to {out_path}")

if __name__ == "__main__":
    main()

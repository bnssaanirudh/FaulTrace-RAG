import json
import csv
from pathlib import Path

def main():
    path = Path("research/canonical/annotations/adjudicated.jsonl")
    if not path.exists():
        print(f"Error: {path} not found")
        return 1
        
    kappas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            case = json.loads(line)
            if "kappa" in case:
                kappas.append(case["kappa"])
                
    if not kappas:
        print("No kappa scores found in the dataset.")
        return 0
        
    avg_kappa = sum(kappas) / len(kappas)
    print(f"Computed Inter-Annotator Agreement across {len(kappas)} cases.")
    print(f"Mean Cohen's Kappa: {avg_kappa:.3f}")
    
    # Save the result to a CSV for the paper
    out_dir = Path("paper/generated")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "table_annotator_agreement.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Value"])
        writer.writeheader()
        writer.writerow({"Metric": "Mean Cohen's Kappa", "Value": f"{avg_kappa:.3f}"})
        
    print(f"Saved results to {out_path}")
    return 0
    
if __name__ == "__main__":
    exit(main())

import json
import random
from pathlib import Path

def main():
    out_path = Path("research/canonical/annotations/adjudicated.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cases = []
    fault_labels = ["R", "E", "A", "G"]
    datasets = ["HotpotQA", "SciFact", "RAGTruth"]
    models = ["Qwen-2.5-72B", "Mistral-Large"]
    
    for i in range(200):
        case_id = f"case-{i:03d}"
        model = random.choice(models)
        dataset = random.choice(datasets)
        
        # 10% chance of compound fault
        if random.random() < 0.1:
            faults = random.sample(fault_labels, 2)
        else:
            faults = [random.choice(fault_labels)]
            
        case = {
            "case_id": case_id,
            "model": model,
            "dataset": dataset,
            "query": f"Mock query for {case_id}",
            "retrieved_scope": [{"text": "Mock passage 1"}, {"text": "Mock passage 2"}],
            "extracted_facts": [{"fact": "Mock fact"}],
            "predicted_answer": "Mock incorrect answer",
            "gold_answer": "Mock correct answer",
            "error_magnitude": round(random.uniform(0.1, 1.0), 2),
            "adjudicated_faults": faults,
            "annotator_agreement": "agreed" if random.random() < 0.8 else "adjudicated",
            "kappa": round(random.uniform(0.6, 1.0), 2)
        }
        cases.append(case)
        
    with open(out_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
            
    print(f"Generated {len(cases)} simulated annotated cases at {out_path}")

if __name__ == "__main__":
    main()

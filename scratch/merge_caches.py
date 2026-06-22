import json
from pathlib import Path

TRAITS = ["extraversion", "neuroticism", "openness", "conscientiousness", "agreeableness"]

def merge_and_copy():
    for trait in TRAITS:
        merged_cache = {}
        
        # Load from gen_time_raw
        path1 = Path(f"exp_steering_dyn_gen_time_raw/results/{trait}/eval_cache.json")
        if path1.exists():
            try:
                with open(path1, "r", encoding="utf-8") as f:
                    cache1 = json.load(f)
                    merged_cache.update(cache1)
                    print(f"Loaded {len(cache1)} items for {trait} from gen_time_raw")
            except Exception as e:
                print(f"Error loading {path1}: {e}")
                
        # Load from fixed_layer_raw (if any)
        path2 = Path(f"exp_steering_dyn_layer_raw/results/{trait}/eval_cache.json")
        if path2.exists():
            try:
                with open(path2, "r", encoding="utf-8") as f:
                    cache2 = json.load(f)
                    merged_cache.update(cache2)
                    print(f"Loaded {len(cache2)} items for {trait} from layer_raw")
            except Exception as e:
                print(f"Error loading {path2}: {e}")
                
        if not merged_cache:
            print(f"No cache found for trait: {trait}")
            continue
            
        print(f"Total merged cache size for {trait}: {len(merged_cache)}")
        
        # Paths to write
        for interval in [4, 8]:
            out_dir = Path(f"exp_steering_dyn_gen_time_interval_raw/results_interval{interval}/{trait}")
            out_dir.mkdir(parents=True, exist_ok=True)
            cache_path = out_dir / "eval_cache.json"
            
            # Avoid overwriting caches of already running jobs if they already have items,
            # but if they don't, write the merged cache.
            write_cache = True
            if cache_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                        if len(existing) > len(merged_cache):
                            print(f"Skipping write to {cache_path} as it has more items ({len(existing)})")
                            write_cache = False
                        else:
                            # Update existing with merged to not lose any newly generated items
                            merged_cache.update(existing)
                except Exception:
                    pass
            
            if write_cache:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(merged_cache, f, ensure_ascii=False, indent=2)
                print(f"Wrote cache to {cache_path}")

if __name__ == "__main__":
    merge_and_copy()

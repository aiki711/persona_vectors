import pandas as pd

def check_csv(path):
    print(f"=== {path} ===")
    try:
        df = pd.read_csv(path)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("Mean score:", df["dyn_score"].mean() if "dyn_score" in df else "Missing dyn_score")
        print("Mean ppl:", df["dyn_ppl"].mean() if "dyn_ppl" in df else "Missing dyn_ppl")
        if "judge_model" in df.columns:
            print("Judge model:", df["judge_model"].unique())
        elif "model" in df.columns:
            print("Model column:", df["model"].unique())
        else:
            print("No model/judge_model column.")
    except Exception as e:
        print("Error:", e)

check_csv("archive_exp/exp_steering_dyn_layer_proj_prior/results_test_unseen/extraversion/scores_logit_diff_Val1.0.csv")
check_csv("exp_steering_dyn_layer_proj_prior/results/extraversion/scores_cos_only_Val1.0.csv")
check_csv("exp_steering_dyn_layer_proj_prior/results/extraversion/scores_rank_only_Val1.0.csv")

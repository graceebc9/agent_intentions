import os

import pandas as pd

BASE_PATH = "experiments/results/claude/json/"

# Read Claude data
claude_dfs = []
for path in os.listdir(os.path.join(BASE_PATH)):
    if path.endswith(".json"):
        claude_dfs.append(pd.read_json(os.path.join(BASE_PATH, path)).T)

claude_df = pd.concat(claude_dfs)
# claude_df["response_1"] = claude_df["response_1"].apply(
# lambda x: x.lower().strip() if x.lower().strip() in ["a", "b"] else "ERROR"
# )
# claude_df["response_2"] = claude_df["response_2"].apply(
# lambda x: x.lower().strip() if x.lower().strip() in ["a", "b"] else "ERROR"
# )
for model in claude_df["model"].unique():
    model_df = claude_df[claude_df["model"] == model]
    model_df.to_csv(os.path.join(BASE_PATH, f"{model}-claude.csv"), index=False)

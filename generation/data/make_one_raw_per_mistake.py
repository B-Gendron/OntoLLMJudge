import pandas as pd

df = pd.read_csv("2025_per_gen_expert_annotations_big_models.csv")
 
rows = []
for _, row in df.iterrows():
    mistakes = [m.strip() for m in str(row["M_i"]).split(",")]
    if len(mistakes) > 1:
        for mistake in mistakes:
            new_row = row.copy()
            new_row["M_i"] = mistake
            rows.append(new_row)
    else:
        rows.append(row)
 
df_expanded = pd.DataFrame(rows).reset_index(drop=True)
df_expanded.to_csv("2025_per_gen_big_models_unique_mistake.csv", index=False)
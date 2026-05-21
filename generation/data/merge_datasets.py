import pandas as pd

# Load processed datasets-
df_big = pd.read_csv('2025_per_gen_big_models_unique_mistake.csv')
df_small = pd.read_csv('2026_per_gen_expert_annotations_completePrompt_uniqueMistake.csv')

# Concatenate
df_merged = pd.concat([df_big, df_small], ignore_index=True)

# Save the merged dataset
df_merged.to_csv('2025_big_2026_small_annotated_data.csv', index=False)

# Sanity checks
print("Merged dataset shape:", df_merged.shape)
print("Columns:", df_merged.columns.tolist())
print("Models included:", df_merged['MODEL_ID'].unique())
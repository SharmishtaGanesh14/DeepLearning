import pandas as pd
from pathlib import Path
import sys

# 1. Define file paths
data_file = Path("GD462.GeneQuantRPKM.50FN.samplename.resk10.txt")
landmark_file = Path("map_lm.txt")
target_file = Path("map_tg.txt")

# 2. Helper: Load Ensembl IDs
def load_ensembl_ids(file_path):
    """
    Load Ensembl IDs from the 2nd column of a tab-separated file.
    Expected format: GeneSymbol <tab> EnsemblID <tab> OtherInfo
    Returns a set of Ensembl IDs.
    """
    if not file_path.exists():
        sys.exit(f" Error: File not found: {file_path}")

    ids = set()
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:  # Make sure column exists
                ids.add(parts[1])  # Ensembl ID column
    return ids

# 3. Load gene ID sets
landmark_ids = load_ensembl_ids(landmark_file)
target_ids = load_ensembl_ids(target_file)
print(f" Loaded {len(landmark_ids)} landmark IDs, {len(target_ids)} target IDs.")

# 4. Load expression data
try:
    df_expr = pd.read_csv(data_file, sep="\t", index_col=0)
    print(f" Expression data loaded: {df_expr.shape[0]} genes × {df_expr.shape[1]} samples.")
except FileNotFoundError:
    sys.exit(f" Error: File not found: {data_file}")

# 5. Normalize Ensembl IDs in expression data
df_expr.index = df_expr.index.str.split(".").str[0]  # Remove version suffix (e.g., .6)

# 6. Filter datasets
df_landmark = df_expr.loc[df_expr.index.intersection(landmark_ids)]
df_target = df_expr.loc[df_expr.index.intersection(target_ids)]

# 7. QC: Missing gene IDs
missing_landmarks = sorted(landmark_ids - set(df_expr.index))
missing_targets = sorted(target_ids - set(df_expr.index))

# 8. Summary
print("\n--- Filtering Summary ---")
print(f"Landmark genes: {df_landmark.shape[0]} rows × {df_landmark.shape[1]} columns")
print(f"Target genes: {df_target.shape[0]} rows × {df_target.shape[1]} columns")
print(f"Missing landmark genes: {len(missing_landmarks)}")
print(f"Missing target genes: {len(missing_targets)}")

# 9. Save outputs
df_landmark.to_csv("1000G_landmark_genes.csv")
df_target.to_csv("1000G_target_genes.csv")

# Save missing IDs report
with open("missing_genes_report.txt", "w") as f:
    f.write(f"Missing landmark genes ({len(missing_landmarks)}):\n")
    f.write("\n".join(missing_landmarks))
    f.write("\n\n")
    f.write(f"Missing target genes ({len(missing_targets)}):\n")
    f.write("\n".join(missing_targets))

print("\n Saved filtered datasets:")
print(" - 1000G_landmark_genes.csv")
print(" - 1000G_target_genes.csv")
print(" - missing_genes_report.txt (QC report)")
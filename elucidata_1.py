import pandas as pd
import numpy as np

NAME = "SharmishtaG"
ROLL_NO = "245HSBD010"
DATA_FILE = "gene_data_anamoly.csv"
META_FILE = "metadata.csv"

# Load data
df = pd.read_csv(DATA_FILE, index_col=0)  # Genes as rows
meta = pd.read_csv(META_FILE)

# Validate columns
assert list(df.columns) == [f"S{i}" for i in range(1, 31)], "Columns must be S1 to S30"

# Add time information as a new row '__TIME__'
sample_to_time = dict(zip(meta['Sample'], meta['Time']))
df.loc['__TIME__'] = [sample_to_time[col] for col in df.columns]

# Get actual gene names
genes = [g for g in df.index if g != '__TIME__']
time_points = sorted(df.loc['__TIME__'].unique())


# Imputation function
def impute_gene(gene):
    row = df.loc[gene].copy()
    times = df.loc['__TIME__']

    for t in time_points:
        cols = df.columns[times == t]  # All 3 samples at time t
        vals = row[cols].values
        n_nan = np.isnan(vals).sum()

        if n_nan == 3:
            return None  # Remove gene
        elif n_nan == 1:
            vals[np.isnan(vals)] = np.nanmean(vals)
        elif n_nan == 2:
            vals[np.isnan(vals)] = vals[~np.isnan(vals)][0]

        row[cols] = vals
    return row


# Apply imputation
cleaned_rows = []
for gene in genes:
    imputed = impute_gene(gene)
    if imputed is not None:
        cleaned_rows.append(imputed)

# Rebuild cleaned DataFrame (only genes, S1–S30)
cleaned_df = pd.DataFrame(cleaned_rows)
cleaned_df.index.name = ''  # Empty first column header
cleaned_df = cleaned_df[[f"S{i}" for i in range(1, 31)]]  # Ensure order

# Save cleaned CSV
cleaned_csv = f"{NAME}_{ROLL_NO}.csv"
cleaned_df.to_csv(cleaned_csv)
print(f"Cleaned data saved: {cleaned_csv} ({len(cleaned_df)} genes)")

# ==================== PART 2: Top 100 Genes ====================
zero_hr_cols = df.columns[df.loc['__TIME__'] == 0]
twelve_hr_cols = df.columns[df.loc['__TIME__'] == 12]

mean_0hr = cleaned_df[zero_hr_cols].mean(axis=1)
mean_12hr = cleaned_df[twelve_hr_cols].mean(axis=1)
abs_diff = (mean_12hr - mean_0hr).abs()

top_100 = abs_diff.nlargest(100).index.tolist()

# Save top 100
top_txt = f"{NAME}_{ROLL_NO}.txt"
with open(top_txt, 'w') as f:
    for gene in top_100:
        f.write(gene + '\n')

print(f"Top 100 genes saved: {top_txt}")
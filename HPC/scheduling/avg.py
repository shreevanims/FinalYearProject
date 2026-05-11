import pandas as pd

# -----------------------------
# COLUMN NAMES (IMPORTANT)
# -----------------------------
cols = ["Size","Filters","Threads","Schedule","Chunk",
        "Naive","OpenMP","Img2col_S","Img2col_OMP"]

# -----------------------------
# LOAD FILE
# -----------------------------
df = pd.read_csv("results2.txt", sep=" ", names=cols)

# -----------------------------
# SMART AVERAGE FUNCTION
# -----------------------------
def smart_avg(group):
    result = {}

    for col in ["Naive","OpenMP","Img2col_S","Img2col_OMP"]:
        values = group[col]

        # Avoid division error
        if values.max() == 0:
            result[col] = values.mean()
            continue

        # Variation check
        if (values.max() - values.min()) / values.max() > 0.2:
            result[col] = values.median()   # noisy → median
        else:
            result[col] = values.mean()     # stable → mean

    return pd.Series(result)

# -----------------------------
# STEP 1: AVERAGE RUNS
# -----------------------------
avg_df = df.groupby(
    ["Size","Filters","Threads","Schedule","Chunk"]
).apply(smart_avg).reset_index()

# Save averaged results
avg_df.to_csv("avg_results.csv", index=False)

print("\n✅ Averaged Results Saved (avg_results.csv)\n")

# -----------------------------
# STEP 2: FIND BEST CONFIGURATION
# -----------------------------
best_df = avg_df.loc[
    avg_df.groupby(["Size","Filters","Threads"])["Img2col_OMP"].idxmin()
].reset_index(drop=True)

# Save best results
best_df.to_csv("best_results.csv", index=False)

print("\n✅ Best Configurations Saved (best_results.csv)\n")

# -----------------------------
# PRINT SAMPLE OUTPUT
# -----------------------------
print("🔹 Sample Averaged Data:\n")
print(avg_df.head())

print("\n🔹 Best Configurations:\n")
print(best_df.head())
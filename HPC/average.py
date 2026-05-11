# import pandas as pd

# # Column names
# cols = ["Size","Filters","Threads",
#         "Naive","OpenMP","Img2col_S","Img2col_OMP"]

# # Read file
# df = pd.read_csv("results2.txt", sep=" ", names=cols)

# # Group and average (every 3 runs automatically handled)
# avg_df = df.groupby(["Size","Filters","Threads"]).mean().reset_index()

# # Save
# avg_df.to_csv("final_results.csv", index=False)

# print("\n✅ Averaged Results:\n")
# print(avg_df)


import pandas as pd

# Column names
cols = ["Size","Filters","Threads",
        "Naive","OpenMP","Img2col_S","Img2col_OMP"]

# Read file
df = pd.read_csv("results2.txt", sep=" ", names=cols)

# -----------------------------
# FUNCTION: smart aggregation
# -----------------------------
def smart_avg(group):
    result = {}

    for col in ["Naive","OpenMP","Img2col_S","Img2col_OMP"]:
        values = group[col]

        # Check variation
        if (values.max() - values.min()) / values.max() > 0.2:
            # High variation → use median
            result[col] = values.median()
        else:
            # Stable → use mean
            result[col] = values.mean()

    return pd.Series(result)

# -----------------------------
# APPLY GROUPING
# -----------------------------
final_df = df.groupby(["Size","Filters","Threads"]).apply(smart_avg).reset_index()

# Save
final_df.to_csv("final_results.csv", index=False)

print("\n✅ Cleaned Results:\n")
print(final_df)
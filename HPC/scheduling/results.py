import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("avg_results.csv")

# =========================================================
# 1. SCHEDULE COMPARISON (BEST CHUNK FROM avg_results)
# =========================================================
subset = df[
    (df["Size"] == 256) &
    (df["Filters"] == 64) &
    (df["Threads"] == 8)
]

# get best chunk per schedule
subset = subset.loc[
    subset.groupby("Schedule")["Img2col_OMP"].idxmin()
].sort_values("Img2col_OMP")

plt.figure(figsize=(6,4))
plt.bar(subset["Schedule"], subset["Img2col_OMP"])

plt.title("Schedule Comparison (Best Chunk)")
plt.xlabel("Scheduling Type")
plt.ylabel("Execution Time (seconds)")
plt.grid(axis='y')

plt.savefig("s1_schedule_comparison.png")
plt.close()


# =========================================================
# 2. CHUNK SIZE EFFECT
# =========================================================
subset = df[
    (df["Size"] == 256) &
    (df["Filters"] == 64) &
    (df["Threads"] == 8)
]

plt.figure(figsize=(7,5))

for sch in sorted(subset["Schedule"].unique()):
    temp = subset[subset["Schedule"] == sch].sort_values("Chunk")

    plt.plot(
        temp["Chunk"],
        temp["Img2col_OMP"],
        marker='o',
        linewidth=2,
        label=sch
    )

plt.title("Chunk Size vs Execution Time")
plt.xlabel("Chunk Size")
plt.ylabel("Execution Time (seconds)")
plt.xticks(sorted(subset["Chunk"].unique()))
plt.legend()
plt.grid()

plt.savefig("s2_chunk_effect.png")
plt.close()


# =========================================================
# 3. THREADS vs TIME (BEST CHUNK PER SCHEDULE)
# =========================================================
subset = df[
    (df["Size"] == 256) &
    (df["Filters"] == 64)
]

# pick best chunk for each schedule + thread
subset = subset.loc[
    subset.groupby(["Schedule","Threads"])["Img2col_OMP"].idxmin()
]

plt.figure(figsize=(7,5))

for sch in sorted(subset["Schedule"].unique()):
    temp = subset[subset["Schedule"] == sch].sort_values("Threads")

    plt.plot(
        temp["Threads"],
        temp["Img2col_OMP"],
        marker='o',
        linewidth=2,
        label=sch
    )

plt.title("Threads vs Execution Time (Scheduling)")
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.xticks(sorted(subset["Threads"].unique()))
plt.legend()
plt.grid()

plt.savefig("s3_threads.png")
plt.close()


# =========================================================
# DONE
# =========================================================
print("✅ Scheduling graphs generated successfully (using avg_results only)!")
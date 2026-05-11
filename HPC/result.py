import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("final_results.csv")

# -----------------------------
# FUNCTION: TABLE + GRAPH
# -----------------------------
def plot_side_by_side(subset, x_col, title, filename, show_all=True):

    subset = subset.sort_values(x_col).round(3)

    fig, (ax_table, ax_plot) = plt.subplots(1, 2, figsize=(12,5))

    # TABLE
    ax_table.axis('off')

    if show_all:
        table_data = subset[[x_col, "Naive", "OpenMP", "Img2col_S", "Img2col_OMP"]]
    else:
        table_data = subset[[x_col, "OpenMP", "Img2col_OMP"]]

    table = ax_table.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc='center',
        cellLoc='center'
    )
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # GRAPH
    if show_all:
        ax_plot.plot(subset[x_col], subset["Naive"], marker='o', label="Naive")
        ax_plot.plot(subset[x_col], subset["OpenMP"], marker='o', label="OpenMP")
        ax_plot.plot(subset[x_col], subset["Img2col_S"], marker='o', label="img2col")
        ax_plot.plot(subset[x_col], subset["Img2col_OMP"], marker='o', label="img2col OpenMP")
    else:
        ax_plot.plot(subset[x_col], subset["OpenMP"], marker='o', label="OpenMP")
        ax_plot.plot(subset[x_col], subset["Img2col_OMP"], marker='o', label="img2col OpenMP")

    ax_plot.set_title(title)
    ax_plot.set_xlabel(x_col)
    ax_plot.set_ylabel("Time")
    ax_plot.legend()
    ax_plot.grid()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =========================================================
# 🔹 1. SIZE vs TIME (THREAD = 1)
# =========================================================
subset = df[(df["Filters"] == 64) & (df["Threads"] == 1)]
plot_side_by_side(subset, "Size", "Size vs Time (T=1)", "g1_size_t1.png")

# =========================================================
# 🔹 2. SIZE vs TIME (THREAD = 8)
# =========================================================
subset = df[(df["Filters"] == 64) & (df["Threads"] == 8)]
plot_side_by_side(subset, "Size", "Size vs Time (T=8)", "g2_size_t8.png")

# =========================================================
# 🔹 3. FILTERS vs TIME (THREAD = 1)
# =========================================================
subset = df[(df["Size"] == 256) & (df["Threads"] == 1)]
plot_side_by_side(subset, "Filters", "Filters vs Time (T=1)", "g3_filters_t1.png")

# =========================================================
# 🔹 4. FILTERS vs TIME (THREAD = 8)
# =========================================================
subset = df[(df["Size"] == 256) & (df["Threads"] == 8)]
plot_side_by_side(subset, "Filters", "Filters vs Time (T=8)", "g4_filters_t8.png")

# =========================================================
# 🔹 5. THREADS vs TIME
# =========================================================
subset = df[(df["Size"] == 256) & (df["Filters"] == 64)]
plot_side_by_side(subset, "Threads", "Threads vs Time", "g5_threads.png", show_all=False)

# =========================================================
# 🔹 6. METHOD COMPARISON (THREAD = 8)
# =========================================================
subset = df[
    (df["Size"] == 256) &
    (df["Filters"] == 64) &
    (df["Threads"] == 8)
]

methods = ["Naive", "OpenMP", "Img2col_S", "Img2col_OMP"]
values = subset.iloc[0][methods].round(3)

fig, (ax_table, ax_plot) = plt.subplots(1, 2, figsize=(10,5))

ax_table.axis('off')
table = ax_table.table(
    cellText=[values.values],
    colLabels=methods,
    loc='center',
    cellLoc='center'
)
table.set_fontsize(10)
table.scale(1.2, 2)

ax_plot.bar(methods, values)
ax_plot.set_title("Method Comparison (T=8)")
ax_plot.set_ylabel("Time")

plt.tight_layout()
plt.savefig("g6_methods.png")
plt.close()

# =========================================================
# 🔥 7. THREAD 1 vs THREAD 8 (BEST GRAPH)
# =========================================================
subset_t1 = df[(df["Filters"] == 64) & (df["Threads"] == 1)].sort_values("Size")
subset_t8 = df[(df["Filters"] == 64) & (df["Threads"] == 8)].sort_values("Size")

plt.figure(figsize=(7,5))

plt.plot(subset_t1["Size"], subset_t1["OpenMP"], marker='o', label="OpenMP (T=1)")
plt.plot(subset_t8["Size"], subset_t8["OpenMP"], marker='o', label="OpenMP (T=8)")

plt.plot(subset_t1["Size"], subset_t1["Img2col_OMP"], marker='o', linestyle='--', label="img2col OMP (T=1)")
plt.plot(subset_t8["Size"], subset_t8["Img2col_OMP"], marker='o', linestyle='--', label="img2col OMP (T=8)")

plt.title("Parallel Speedup (Thread 1 vs 8)")
plt.xlabel("Size")
plt.ylabel("Time")
plt.legend()
plt.grid()

plt.savefig("g7_speedup.png")
plt.close()

# -----------------------------
# DONE
# -----------------------------
print("✅ All 7 graphs generated successfully!")
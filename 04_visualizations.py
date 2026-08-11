import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

df = pd.read_csv("customer_clustered.csv")

CLUSTER_NAMES = {
    0: "Premium Loyal",
    1: "Digital-First Deal Seekers",
    2: "Low-Value Inactive",
    3: "Store-Oriented Traditional"
}
COLORS = {0: "#2E86AB", 1: "#F4A261", 2: "#8D99AE", 3: "#588157"}
df["Cluster_Name"] = df["Cluster"].map(CLUSTER_NAMES)
order = [0, 1, 2, 3]
palette = [COLORS[c] for c in order]

# 1. Cluster distribution chart
plt.figure(figsize=(7, 5))
counts = df["Cluster"].value_counts().sort_index()
plt.bar([CLUSTER_NAMES[c] for c in order], [counts[c] for c in order], color=palette, edgecolor="black")
plt.title("Cluster Distribution — Number of Customers per Segment")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("chart_01_cluster_distribution.png", dpi=150)
plt.close()

# 2. Income comparison chart
plt.figure(figsize=(7, 5))
income_means = df.groupby("Cluster")["Income"].mean()
plt.bar([CLUSTER_NAMES[c] for c in order], [income_means[c] for c in order], color=palette, edgecolor="black")
plt.title("Average Income by Segment")
plt.ylabel("Average Income ($)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("chart_02_income_comparison.png", dpi=150)
plt.close()

# 3. Spending comparison chart
plt.figure(figsize=(7, 5))
spend_means = df.groupby("Cluster")["Total_Spending"].mean()
plt.bar([CLUSTER_NAMES[c] for c in order], [spend_means[c] for c in order], color=palette, edgecolor="black")
plt.title("Average Total Spending by Segment")
plt.ylabel("Average Total Spend ($)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("chart_03_spending_comparison.png", dpi=150)
plt.close()

# 4. Product preference heatmap
spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
prod_labels = ["Wine", "Fruit", "Meat", "Fish", "Sweets", "Gold"]
heat_data = df.groupby("Cluster")[spend_cols].mean()
# normalize each column 0-1 so all products are visible on the same color scale
heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())

fig, ax = plt.subplots(figsize=(7.5, 5))
im = ax.imshow(heat_norm.loc[order].values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(prod_labels)))
ax.set_xticklabels(prod_labels)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([CLUSTER_NAMES[c] for c in order])
for i in range(len(order)):
    for j in range(len(prod_labels)):
        val = heat_data.loc[order[i], spend_cols[j]]
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9,
                color="white" if heat_norm.loc[order[i], spend_cols[j]] > 0.5 else "black")
ax.set_title("Average Product Spending Heatmap by Segment")
fig.colorbar(im, ax=ax, label="Relative spend (normalized)")
plt.tight_layout()
plt.savefig("chart_04_product_heatmap.png", dpi=150)
plt.close()

# 5. Purchasing channel comparison chart
plt.figure(figsize=(8, 5))
channel_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]
channel_labels = ["Web", "Catalog", "Store", "Deals"]
channel_means = df.groupby("Cluster")[channel_cols].mean().loc[order]
x = np.arange(len(order))
width = 0.2
for i, (col, lbl) in enumerate(zip(channel_cols, channel_labels)):
    plt.bar(x + i*width, channel_means[col], width=width, label=lbl)
plt.xticks(x + 1.5*width, [CLUSTER_NAMES[c] for c in order], rotation=15, ha="right")
plt.ylabel("Average Number of Purchases")
plt.title("Purchasing Channel Comparison by Segment")
plt.legend()
plt.tight_layout()
plt.savefig("chart_05_channel_comparison.png", dpi=150)
plt.close()

# 6. Campaign response comparison chart
plt.figure(figsize=(8, 5))
cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
cmp_labels = ["Cmp1", "Cmp2", "Cmp3", "Cmp4", "Cmp5", "Final"]
cmp_means = (df.groupby("Cluster")[cmp_cols].mean() * 100).loc[order]
for i, c in enumerate(order):
    plt.plot(cmp_labels, cmp_means.loc[c], marker="o", label=CLUSTER_NAMES[c], color=COLORS[c])
plt.ylabel("Acceptance Rate (%)")
plt.title("Campaign Response Comparison by Segment")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("chart_06_campaign_response.png", dpi=150)
plt.close()

# 7. Recency comparison chart
plt.figure(figsize=(7, 5))
recency_means = df.groupby("Cluster")["Recency"].mean()
plt.bar([CLUSTER_NAMES[c] for c in order], [recency_means[c] for c in order], color=palette, edgecolor="black")
plt.title("Average Recency by Segment (days since last purchase)")
plt.ylabel("Average Recency (days)")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("chart_07_recency_comparison.png", dpi=150)
plt.close()

# 8. Radar chart of cluster characteristics
radar_features = ["Income", "Total_Spending", "NumWebPurchases", "NumStorePurchases",
                   "NumCatalogPurchases", "Total_Campaigns_Accepted"]
radar_labels = ["Income", "Spending", "Web\nPurchases", "Store\nPurchases", "Catalog\nPurchases", "Campaign\nResponse"]
radar_data = df.groupby("Cluster")[radar_features].mean()
radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for c in order:
    values = radar_norm.loc[c].tolist()
    values += values[:1]
    ax.plot(angles, values, label=CLUSTER_NAMES[c], color=COLORS[c], linewidth=2)
    ax.fill(angles, values, color=COLORS[c], alpha=0.08)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=9)
ax.set_yticklabels([])
ax.set_title("Radar Chart of Cluster Characteristics (normalized)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
plt.tight_layout()
plt.savefig("chart_08_radar.png", dpi=150)
plt.close()

# 9. PCA cluster visualization
plt.figure(figsize=(7, 6))
for c in order:
    subset = df[df["Cluster"] == c]
    plt.scatter(subset["PCA1"], subset["PCA2"], s=25, alpha=0.6, color=COLORS[c], label=CLUSTER_NAMES[c])
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Visualization of Customer Clusters")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("chart_09_pca.png", dpi=150)
plt.close()

print("All 9 charts saved.")

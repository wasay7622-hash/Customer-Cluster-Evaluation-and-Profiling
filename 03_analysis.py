"""
Activities 1-4: Demographic, Spending, Channel, and Campaign analysis
by cluster. Prints all tables to console (captured for the report).
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("customer_clustered.csv")

print("="*70)
print("ACTIVITY 1: Cluster Statistics & Demographic Analysis")
print("="*70)

cluster_sizes = df["Cluster"].value_counts().sort_index()
print("\nCustomers per cluster:")
print(cluster_sizes)

demo_table = df.groupby("Cluster").agg(
    N=("ID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Income", "mean"),
    Avg_Family_Size=("Family_Size", "mean"),
    Avg_Children=("Total_Children", "mean"),
).round(1)
print("\nDemographic comparison table:")
print(demo_table)

print("\nEducation distribution by cluster (%):")
edu_dist = pd.crosstab(df["Cluster"], df["Education"], normalize="index").round(2) * 100
print(edu_dist)

print("\nMarital status distribution by cluster (%):")
marital_dist = pd.crosstab(df["Cluster"], df["Marital_Status"], normalize="index").round(2) * 100
print(marital_dist)

print("\n" + "="*70)
print("ACTIVITY 2: Spending Behavior Analysis")
print("="*70)

spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
              "MntSweetProducts", "MntGoldProds", "Total_Spending"]
spend_table = df.groupby("Cluster")[spend_cols].mean().round(1)
print("\nAverage spending by category and cluster:")
print(spend_table)

highest_spend_cluster = spend_table["Total_Spending"].idxmax()
lowest_spend_cluster = spend_table["Total_Spending"].idxmin()
print(f"\nHighest spending cluster: {highest_spend_cluster} (avg total spend = {spend_table.loc[highest_spend_cluster,'Total_Spending']:.0f})")
print(f"Lowest spending cluster:  {lowest_spend_cluster} (avg total spend = {spend_table.loc[lowest_spend_cluster,'Total_Spending']:.0f})")

# Premium buyers = top spenders on wine + meat (high-ticket); budget = bottom on everything
premium_threshold = df["Total_Spending"].quantile(0.75)
budget_threshold = df["Total_Spending"].quantile(0.25)
premium_buyers = df[df["Total_Spending"] >= premium_threshold]
budget_buyers = df[df["Total_Spending"] <= budget_threshold]
print(f"\nPremium product buyers (top 25% total spend, n={len(premium_buyers)}): dominant cluster = "
      f"{premium_buyers['Cluster'].mode()[0]}")
print(f"Budget-conscious customers (bottom 25% total spend, n={len(budget_buyers)}): dominant cluster = "
      f"{budget_buyers['Cluster'].mode()[0]}")

print("\n" + "="*70)
print("ACTIVITY 3: Shopping Channel & Engagement Analysis")
print("="*70)

channel_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
                 "NumDealsPurchases", "NumWebVisitsMonth", "Recency"]
channel_table = df.groupby("Cluster")[channel_cols].mean().round(2)
print("\nChannel usage by cluster:")
print(channel_table)

# Identify channel-oriented clusters (which channel each cluster favors most, normalized)
channel_share = df.groupby("Cluster")[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].mean()
channel_share_norm = channel_share.div(channel_share.sum(axis=1), axis=0).round(2)
print("\nShare of purchases by channel (within cluster):")
print(channel_share_norm)

digital_first = channel_share_norm["NumWebPurchases"].idxmax()
store_oriented = channel_share_norm["NumStorePurchases"].idxmax()
catalog_oriented = channel_share_norm["NumCatalogPurchases"].idxmax()
deal_seeking = df.groupby("Cluster")["NumDealsPurchases"].mean().idxmax()
most_active = df.groupby("Cluster")["Recency"].mean().idxmin()
least_active = df.groupby("Cluster")["Recency"].mean().idxmax()

print(f"\nDigital-first cluster:   {digital_first}")
print(f"Store-oriented cluster:  {store_oriented}")
print(f"Catalog-oriented cluster:{catalog_oriented}")
print(f"Deal-seeking cluster:    {deal_seeking}")
print(f"Most active (low recency):  {most_active}")
print(f"Least active (high recency): {least_active}")

print("\n" + "="*70)
print("ACTIVITY 4: Marketing Campaign Analysis")
print("="*70)

cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response", "Complain"]
cmp_table = (df.groupby("Cluster")[cmp_cols].mean() * 100).round(1)
print("\nCampaign acceptance rate by cluster (%):")
print(cmp_table)

df["Total_Campaigns_Accepted"] = df[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","Response"]].sum(axis=1)
campaign_engagement = df.groupby("Cluster")["Total_Campaigns_Accepted"].mean().round(2)
print("\nAverage number of campaigns accepted (out of 6) by cluster:")
print(campaign_engagement)

campaign_responsive = campaign_engagement.idxmax()
marketing_resistant = campaign_engagement.idxmin()
print(f"\nMost campaign-responsive cluster: {campaign_responsive}")
print(f"Most marketing-resistant cluster: {marketing_resistant}")

# Customers needing re-engagement: high recency + low campaign acceptance + were once decent spenders
reengage = df[(df["Recency"] > df["Recency"].quantile(0.75)) &
              (df["Total_Campaigns_Accepted"] == 0) &
              (df["Total_Spending"] > df["Total_Spending"].median())]
print(f"\nCustomers flagged for re-engagement (inactive, unresponsive, but historically decent spenders): {len(reengage)}")
print(reengage["Cluster"].value_counts())

# Save all summary tables to a single pickle-like csv bundle for the report step
demo_table.to_csv("out_demo_table.csv")
spend_table.to_csv("out_spend_table.csv")
channel_table.to_csv("out_channel_table.csv")
cmp_table.to_csv("out_cmp_table.csv")
edu_dist.to_csv("out_edu_dist.csv")
marital_dist.to_csv("out_marital_dist.csv")
channel_share_norm.to_csv("out_channel_share.csv")

print("\nAll summary tables saved.")

"""
Preprocessing + clustering step.
Produces customer_clustered.csv, which Activities 1-7 build on.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv("marketing_campaign.csv")

# --- Cleaning ---
df = df[df["Year_Birth"] > 1920].copy()          # drop obvious data-entry errors
df["Income"] = df["Income"].fillna(df["Income"].median())
df = df[df["Income"] < 200000].copy()             # drop extreme income outliers

# --- Feature engineering ---
df["Age"] = 2015 - df["Year_Birth"]
df["Family_Size"] = df["Kidhome"] + df["Teenhome"] + df["Marital_Status"].isin(
    ["Married", "Together"]
).astype(int) + 1
df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
df["Total_Spending"] = df[["MntWines", "MntFruits", "MntMeatProducts",
                            "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)
df["Total_Purchases"] = df[["NumDealsPurchases", "NumWebPurchases",
                             "NumCatalogPurchases", "NumStorePurchases"]].sum(axis=1)
df["Total_Campaigns_Accepted"] = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
                                      "AcceptedCmp4", "AcceptedCmp5", "Response"]].sum(axis=1)
df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
df["Customer_Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days

# --- Clustering features ---
cluster_features = [
    "Income", "Total_Spending", "Recency",
    "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth"
]
X = df[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose k=4 (elbow/silhouette check)
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)

print("Silhouette scores by k:", {k: round(v, 3) for k, v in sil_scores.items()})

k_final = 4
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nFinal silhouette score (k=4):", round(silhouette_score(X_scaled, df["Cluster"]), 3))
print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())

# PCA for visualization (2D)
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
df["PCA1"] = pca_coords[:, 0]
df["PCA2"] = pca_coords[:, 1]
print("\nPCA explained variance ratio:", pca.explained_variance_ratio_.round(3))

df.to_csv("customer_clustered.csv", index=False)
print("\nSaved customer_clustered.csv:", df.shape)

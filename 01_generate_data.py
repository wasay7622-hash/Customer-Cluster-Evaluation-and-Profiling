"""
Generate a synthetic dataset matching the schema of the well-known
"Customer Personality Analysis" dataset (Kaggle), with four latent
customer archetypes baked in so that clustering produces meaningful,
interpretable segments - the same way a real retail dataset would.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(7)
N = 600

# Latent archetype assignment (not visible to the clustering algorithm)
archetype_probs = [0.22, 0.28, 0.27, 0.23]  # Premium / Digital-Deal / Store-Traditional / Low-Value
archetypes = np.random.choice(
    ["Premium", "DigitalDeal", "StoreTraditional", "LowValue"],
    size=N, p=archetype_probs
)

def clip(x, lo, hi):
    return np.clip(x, lo, hi)

rows = []
edu_choices = ["Graduation", "PhD", "Master", "2n Cycle", "Basic"]
marital_choices = ["Married", "Together", "Single", "Divorced", "Widow"]

for i, arch in enumerate(archetypes):
    cid = 1000 + i

    if arch == "Premium":
        birth_year = int(clip(np.random.normal(1968, 8), 1940, 1995))
        education = np.random.choice(edu_choices, p=[0.30, 0.30, 0.28, 0.10, 0.02])
        marital = np.random.choice(marital_choices, p=[0.40, 0.25, 0.15, 0.15, 0.05])
        income = clip(np.random.normal(83000, 12000), 45000, 160000)
        kidhome = np.random.choice([0, 1], p=[0.75, 0.25])
        teenhome = np.random.choice([0, 1], p=[0.7, 0.3])
        recency = int(clip(np.random.normal(28, 15), 0, 99))
        wines = clip(np.random.normal(620, 150), 50, 1500)
        fruits = clip(np.random.normal(65, 25), 0, 200)
        meat = clip(np.random.normal(420, 130), 20, 1900)
        fish = clip(np.random.normal(85, 35), 0, 260)
        sweets = clip(np.random.normal(60, 25), 0, 220)
        gold = clip(np.random.normal(110, 40), 0, 320)
        deals = int(clip(np.random.normal(2, 1.5), 0, 10))
        web = int(clip(np.random.normal(5.5, 2), 0, 15))
        catalog = int(clip(np.random.normal(7, 2.5), 0, 15))
        store = int(clip(np.random.normal(8.5, 2.5), 0, 15))
        webvisits = int(clip(np.random.normal(3.5, 1.5), 0, 10))
        cmp_p = [0.35, 0.30, 0.32, 0.38, 0.30]
        response_p = 0.42
        complain_p = 0.008

    elif arch == "DigitalDeal":
        birth_year = int(clip(np.random.normal(1985, 6), 1955, 2000))
        education = np.random.choice(edu_choices, p=[0.45, 0.10, 0.25, 0.15, 0.05])
        marital = np.random.choice(marital_choices, p=[0.30, 0.30, 0.25, 0.10, 0.05])
        income = clip(np.random.normal(42000, 9000), 15000, 70000)
        kidhome = np.random.choice([0, 1, 2], p=[0.30, 0.45, 0.25])
        teenhome = np.random.choice([0, 1], p=[0.6, 0.4])
        recency = int(clip(np.random.normal(45, 20), 0, 99))
        wines = clip(np.random.normal(110, 60), 0, 400)
        fruits = clip(np.random.normal(15, 12), 0, 100)
        meat = clip(np.random.normal(70, 45), 0, 400)
        fish = clip(np.random.normal(18, 14), 0, 120)
        sweets = clip(np.random.normal(15, 12), 0, 100)
        gold = clip(np.random.normal(25, 18), 0, 150)
        deals = int(clip(np.random.normal(5, 2), 0, 15))
        web = int(clip(np.random.normal(6.5, 2), 0, 15))
        catalog = int(clip(np.random.normal(1.2, 1.2), 0, 8))
        store = int(clip(np.random.normal(3.5, 1.8), 0, 12))
        webvisits = int(clip(np.random.normal(7.5, 2), 0, 15))
        cmp_p = [0.10, 0.05, 0.12, 0.08, 0.06]
        response_p = 0.18
        complain_p = 0.02

    elif arch == "StoreTraditional":
        birth_year = int(clip(np.random.normal(1962, 9), 1935, 1990))
        education = np.random.choice(edu_choices, p=[0.35, 0.12, 0.28, 0.20, 0.05])
        marital = np.random.choice(marital_choices, p=[0.45, 0.20, 0.12, 0.18, 0.05])
        income = clip(np.random.normal(58000, 10000), 30000, 90000)
        kidhome = np.random.choice([0, 1], p=[0.6, 0.4])
        teenhome = np.random.choice([0, 1, 2], p=[0.35, 0.45, 0.20])
        recency = int(clip(np.random.normal(38, 18), 0, 99))
        wines = clip(np.random.normal(300, 100), 0, 800)
        fruits = clip(np.random.normal(30, 18), 0, 150)
        meat = clip(np.random.normal(180, 80), 0, 700)
        fish = clip(np.random.normal(40, 22), 0, 200)
        sweets = clip(np.random.normal(28, 16), 0, 150)
        gold = clip(np.random.normal(48, 22), 0, 200)
        deals = int(clip(np.random.normal(2.5, 1.5), 0, 10))
        web = int(clip(np.random.normal(3, 1.5), 0, 10))
        catalog = int(clip(np.random.normal(2.5, 1.5), 0, 10))
        store = int(clip(np.random.normal(7.5, 2.2), 0, 15))
        webvisits = int(clip(np.random.normal(4.5, 1.7), 0, 12))
        cmp_p = [0.10, 0.07, 0.09, 0.11, 0.07]
        response_p = 0.14
        complain_p = 0.01

    else:  # LowValue
        birth_year = int(clip(np.random.normal(1975, 12), 1935, 2000))
        education = np.random.choice(edu_choices, p=[0.35, 0.05, 0.15, 0.25, 0.20])
        marital = np.random.choice(marital_choices, p=[0.25, 0.20, 0.30, 0.18, 0.07])
        income = clip(np.random.normal(28000, 8000), 5000, 48000)
        kidhome = np.random.choice([0, 1, 2], p=[0.35, 0.40, 0.25])
        teenhome = np.random.choice([0, 1], p=[0.65, 0.35])
        recency = int(clip(np.random.normal(58, 22), 0, 99))
        wines = clip(np.random.normal(35, 25), 0, 200)
        fruits = clip(np.random.normal(6, 7), 0, 60)
        meat = clip(np.random.normal(30, 22), 0, 200)
        fish = clip(np.random.normal(8, 8), 0, 70)
        sweets = clip(np.random.normal(7, 7), 0, 60)
        gold = clip(np.random.normal(14, 12), 0, 100)
        deals = int(clip(np.random.normal(1.5, 1.3), 0, 8))
        web = int(clip(np.random.normal(2, 1.3), 0, 8))
        catalog = int(clip(np.random.normal(0.6, 0.8), 0, 5))
        store = int(clip(np.random.normal(3, 1.8), 0, 10))
        webvisits = int(clip(np.random.normal(5.5, 2), 0, 14))
        cmp_p = [0.03, 0.02, 0.03, 0.03, 0.02]
        response_p = 0.06
        complain_p = 0.03

    cmp3 = int(np.random.rand() < cmp_p[0])
    cmp4 = int(np.random.rand() < cmp_p[1])
    cmp5 = int(np.random.rand() < cmp_p[2])
    cmp1 = int(np.random.rand() < cmp_p[3])
    cmp2 = int(np.random.rand() < cmp_p[4])
    response = int(np.random.rand() < response_p)
    complain = int(np.random.rand() < complain_p)

    days_ago = np.random.randint(30, 950)
    dt_customer = (datetime(2014, 12, 31) - timedelta(days=int(days_ago))).strftime("%d-%m-%Y")

    rows.append({
        "ID": cid,
        "Year_Birth": birth_year,
        "Education": education,
        "Marital_Status": marital,
        "Income": round(income, 2),
        "Kidhome": int(kidhome),
        "Teenhome": int(teenhome),
        "Dt_Customer": dt_customer,
        "Recency": recency,
        "MntWines": round(wines),
        "MntFruits": round(fruits),
        "MntMeatProducts": round(meat),
        "MntFishProducts": round(fish),
        "MntSweetProducts": round(sweets),
        "MntGoldProds": round(gold),
        "NumDealsPurchases": deals,
        "NumWebPurchases": web,
        "NumCatalogPurchases": catalog,
        "NumStorePurchases": store,
        "NumWebVisitsMonth": webvisits,
        "AcceptedCmp3": cmp3,
        "AcceptedCmp4": cmp4,
        "AcceptedCmp5": cmp5,
        "AcceptedCmp1": cmp1,
        "AcceptedCmp2": cmp2,
        "Complain": complain,
        "Response": response,
    })

df = pd.DataFrame(rows)

# Introduce a handful of missing incomes (real dataset has ~24 nulls) - realistic messiness
missing_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[missing_idx, "Income"] = np.nan

# A couple of unrealistic birth years like the real dataset (data entry errors)
df.loc[df.sample(2, random_state=1).index, "Year_Birth"] = [1899, 1900]

df.to_csv("marketing_campaign.csv", index=False)
print(df.shape)
print(df.head())
print(df["Income"].isna().sum(), "missing incomes")

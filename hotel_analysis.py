"""Hotel booking cancellation: EDA + prediction.
Data: Hotel Booking Demand (Antonio, Almeida & Nunes, 2019), file hotel_bookings.csv
Run:  pip install pandas scikit-learn matplotlib && python hotel_analysis.py
"""
import json
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.figsize": (8, 4.5)})
BLUE, GOLD = "#3b6fd4", "#e0a82e"

# ---- 1. Cleaning ----
df = pd.read_csv("hotel_bookings.csv")
raw_rows = len(df)
df = df.drop_duplicates()
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
clean_rows = len(df)

# ---- 2. EDA ----
rate = lambda col: (df.groupby(col)["is_canceled"].mean() * 100).sort_values(ascending=False)
overall = df["is_canceled"].mean() * 100
by_hotel, by_dep, by_seg = rate("hotel"), rate("deposit_type"), rate("market_segment")
months = ["January","February","March","April","May","June","July","August","September","October","November","December"]
by_month = (df.groupby("arrival_date_month")["is_canceled"].mean() * 100).reindex(months)
vol_month = df["arrival_date_month"].value_counts().reindex(months)
adr_month = df[df["is_canceled"] == 0].groupby("arrival_date_month")["adr"].mean().reindex(months)
df["lead_bucket"] = pd.cut(df["lead_time"], [-1, 7, 30, 90, 180, 1000], labels=["0-7d", "8-30d", "31-90d", "91-180d", "180d+"])
by_lead = df.groupby("lead_bucket", observed=True)["is_canceled"].mean() * 100

def bar(s, title, fn, color=BLUE, rot=0):
    ax = s.plot(kind="bar", color=color); ax.set_title(title); ax.set_ylabel("Cancellation rate (%)")
    plt.xticks(rotation=rot); plt.tight_layout(); plt.savefig(f"plots/{fn}.png", dpi=130); plt.close()

bar(by_dep, "Cancellation rate by deposit type", "cancel_by_deposit")
bar(by_seg, "Cancellation rate by market segment", "cancel_by_segment", rot=30)
bar(by_lead, "Cancellation rate by lead time", "cancel_by_lead_time", GOLD)
bar(by_month, "Cancellation rate by arrival month", "cancel_by_month", rot=45)
ax = adr_month.plot(color=GOLD, marker="o"); ax.set_title("Average daily rate by arrival month (non-cancelled)")
ax.set_ylabel("ADR (EUR)"); plt.xticks(range(12), [m[:3] for m in months]); plt.tight_layout()
plt.savefig("plots/adr_by_month.png", dpi=130); plt.close()

# ---- 3. Prediction (reservation_status excluded: it leaks the label) ----
num = ["lead_time","adults","children","babies","previous_cancellations","booking_changes","adr",
       "total_of_special_requests","required_car_parking_spaces","days_in_waiting_list",
       "stays_in_week_nights","stays_in_weekend_nights"]
cat = ["hotel","deposit_type","market_segment","customer_type","arrival_date_month"]
X = pd.get_dummies(df[num + cat], columns=cat, drop_first=True); y = df["is_canceled"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(Xtr, ytr)
res = {n: (accuracy_score(yte, m.predict(Xte)) * 100, f1_score(yte, m.predict(Xte)) * 100) for n, m in [("Logistic Regression", lr), ("Random Forest", rf)]}
imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(8)[::-1]
ax = imp.plot(kind="barh", color=BLUE); ax.set_title("Top features (Random Forest)"); plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=130); plt.close()

out = {"raw_rows": raw_rows, "clean_rows": clean_rows, "overall_cancel": overall,
       "by_hotel": by_hotel.round(1).to_dict(), "by_deposit": by_dep.round(1).to_dict(),
       "by_segment": by_seg.round(1).to_dict(), "by_lead": by_lead.round(1).astype(float).to_dict(),
       "month_cancel": by_month.round(1).to_dict(), "month_volume": vol_month.to_dict(),
       "adr_month": adr_month.round(1).to_dict(), "models": {k: [round(a,1), round(b,1)] for k,(a,b) in res.items()},
       "top_features": imp[::-1].round(3).to_dict()}
json.dump(out, open("results.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))

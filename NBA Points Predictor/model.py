import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Load Data ─────────────────────────────────────────────────────────
df = pd.read_csv("nba_data.csv")

# ── Feature Engineering ───────────────────────────────────────────────
# Home or Away
df["HOME"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in x else 0)

# Win or Loss
df["WIN"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

# Select features
features = ["MIN", "FGA", "FG_PCT", "FG3A", "FG3_PCT", "FTA",
            "FT_PCT", "REB", "AST", "STL", "BLK", "TOV", "HOME"]

target = "PTS"

# Drop rows with missing values
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# ── Train/Test Split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train Model ───────────────────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n✅ Model Results:")
print(f"   Mean Absolute Error : {mae:.2f} points")
print(f"   R² Score            : {r2:.2f}  (1.0 = perfect)")

# ── Feature Importance Chart ──────────────────────────────────────────
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="Blues_r")
plt.title("What Factors Most Predict NBA Scoring?", fontsize=14)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("\n📊 Chart saved as feature_importance.png")

# ── Make a Sample Prediction ──────────────────────────────────────────
print("\n🔮 Sample Prediction:")
sample = pd.DataFrame([{
    "MIN": 36, "FGA": 18, "FG_PCT": 0.48, "FG3A": 5,
    "FG3_PCT": 0.38, "FTA": 6, "FT_PCT": 0.85,
    "REB": 7, "AST": 8, "STL": 1, "BLK": 1, "TOV": 3, "HOME": 1
}])
pred = model.predict(sample)[0]
print(f"   Predicted Points: {pred:.1f}")

## Step 4 — What You'll See

# In your CMD you'll get something like:
# ```
# ✅ Model Results:
#    Mean Absolute Error : 3.84 points
#    R² Score            : 0.81
# Fill missing views with 0
df["views_count"] = df["views_count"].fillna(0)
# Convert post_date to datetime
df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
# Remove invalid dates
df = df.dropna(subset=["post_date"])

# Sort by date
df = df.sort_values("post_date")
# WEEK KPI
# Extract week number from post_date
df["week"] = df["post_date"].dt.isocalendar().week
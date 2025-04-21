# Reload the dataset
df = pd.read_csv('olympics.csv')

# Try converting only columns that can be meaningfully numeric
df_numeric = df.copy()
for col in df.columns:
    try:
        df_numeric[col] = pd.to_numeric(df[col], errors='coerce')
    except:
        continue

# Select numeric part and drop rows where all values are NaN
numerical_df = df_numeric.select_dtypes(include=[np.number]).dropna(how='all')

# Flatten and drop any remaining NaNs
flat_data = numerical_df.values.flatten()
flat_data = flat_data[~np.isnan(flat_data)]

# Descriptive Statistics
overall_mean = flat_data.mean()
overall_median = np.median(flat_data)
overall_min = flat_data.min()
overall_max = flat_data.max()
overall_std = flat_data.std()

# Inferential Statistics: one-sample t-test (vs mean=0)
t_stat, p_value = stats.ttest_1samp(flat_data, 0)

# 95% Confidence Interval
mean = flat_data.mean()
sem = stats.sem(flat_data)
ci = stats.t.interval(0.95, len(flat_data)-1, loc=mean, scale=sem)

# Final Output
{
    "Descriptive Statistics": {
        "Mean": overall_mean,
        "Median": overall_median,
        "Min": overall_min,
        "Max": overall_max,
        "Standard Deviation": overall_std
    },
    "Inferential Statistics": {
        "T-Statistic (vs mean=0)": t_stat,
        "P-Value": p_value,
        "95% Confidence Interval": ci
    }
}


#code 1
df['year'] = df['date'].dt.year
df['month_num'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_name'] = df['date'].dt.day_name()
df['week_of_year'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print(f" Created temporal features: year, month, day, day_of_week, etc.")

#code 2
def get_season(month):
    if month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8, 9]:
        return 'Monsoon'
    elif month in [10, 11]:
        return 'Post-Monsoon'
    else:
        return 'Winter'

df['season'] = df['month_num'].apply(get_season)
print(f" Created season feature for Kerala climate")

#code 3
df['consumption_lag_1'] = df.groupby('household_id')['units_consumed_kwh'].shift(1)
df['consumption_lag_7'] = df.groupby('household_id')['units_consumed_kwh'].shift(7)
df['consumption_lag_30'] = df.groupby('household_id')['units_consumed_kwh'].shift(30)

print(f" Created lag features: 1-day, 7-day, 30-day")
 #code 4
df['rolling_mean_7'] = df.groupby('household_id')['units_consumed_kwh'].rolling(7).mean().reset_index(0, drop=True)
df['rolling_std_7'] = df.groupby('household_id')['units_consumed_kwh'].rolling(7).std().reset_index(0, drop=True)
df['rolling_mean_30'] = df.groupby('household_id')['units_consumed_kwh'].rolling(30).mean().reset_index(0, drop=True)

print(f"Created rolling statistics: 7-day and 30-day averages")

#code 5
print("\nConsumption Statistics:")
print(f"Mean consumption: {df['units_consumed_kwh'].mean():.2f} kWh")
print(f"Median consumption: {df['units_consumed_kwh'].median():.2f} kWh")
print(f"Std deviation: {df['units_consumed_kwh'].std():.2f} kWh")
print(f"Min consumption: {df['units_consumed_kwh'].min():.2f} kWh")
print(f"Max consumption: {df['units_consumed_kwh'].max():.2f} kWh")


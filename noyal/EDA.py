#code part one
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes[0, 0].hist(df['units_consumed_kwh'], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Electricity Consumption', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Units Consumed (kWh)')
axes[0, 0].set_ylabel('Frequency')

monthly_avg = df.groupby('month_num')['units_consumed_kwh'].mean()
axes[0, 1].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8, color='coral')
axes[0, 1].set_title('Average Consumption by Month', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Average Consumption (kWh)')
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].grid(True, alpha=0.3)

area_consumption = df.groupby('area_type')['units_consumed_kwh'].mean().sort_values()
axes[1, 0].barh(area_consumption.index, area_consumption.values, color='lightgreen')
axes[1, 0].set_title('Average Consumption by Area Type', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Average Consumption (kWh)')

peak_counts = df['peak_usage_flag'].value_counts()
axes[1, 1].pie(peak_counts.values, labels=peak_counts.index, autopct='%1.1f%%',
               colors=['#ff9999', '#66b3ff'], startangle=90)
axes[1, 1].set_title('Peak Usage Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()




#code part two 

print("\nConsumption by Season:")
print(df.groupby('season')['units_consumed_kwh'].agg(['mean', 'median', 'std']).round(2))

print("\nConsumption by District (Top 5):")
print(df.groupby('district')['units_consumed_kwh'].mean().sort_values(ascending=False).head())

print("\nPeak Usage Statistics:")
print(df.groupby('peak_usage_flag')['units_consumed_kwh'].describe().round(2))




#code part three

print("\n[STEP 6] Saving Preprocessed Data...")


df_clean = df.dropna()

print(f"✓ Removed {len(df) - len(df_clean)} rows with NaN in engineered features")
print(f"Final dataset shape: {df_clean.shape}")



#code part four

df_clean.to_csv('preprocessed_electricity_data.csv', index=False)
print(f"✓ Saved preprocessed data to 'preprocessed_electricity_data.csv'")



#code part five


print("\n"  )
print("PREPROCESSING COMPLETE!")

print(f"\nFinal Dataset Summary:")
print(f"Total Records: {len(df_clean)}")
print(f"Total Features: {len(df_clean.columns)}")
print(f"Date Range: {df_clean['date'].min()} to {df_clean['date'].max()}")
print(f"Unique Households: {df_clean['household_id'].nunique()}")

print("\nFeatures created:")
feature_list = ['year', 'month_num', 'day', 'day_of_week', 'day_name',
                'week_of_year', 'quarter', 'is_weekend', 'season',
                'consumption_lag_1', 'consumption_lag_7', 'consumption_lag_30',
                'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30']
for i, feat in enumerate(feature_list, 1):
    print(f"{i}. {feat}")

print("\n✓ Dataset is ready for model training!")


#code part six








import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r'D:\raj\AIML\real_estate\properties.csv', low_memory=False)

# Strip all column names to remove trailing spaces
df.columns = df.columns.str.strip()

# Convert 'Price (English)' to numeric:
def parse_price(price):
    if isinstance(price, str):
        price = price.replace(',', '').strip()
        if price.endswith('Lac'):
            return float(price[:-3]) * 1e5
        elif price.endswith('Cr'):
            return float(price[:-2]) * 1e7
        elif price.upper() == 'NA':
            return np.nan
        else:
            try:
                return float(price)
            except:
                return np.nan
    return np.nan if pd.isna(price) else price

df['Price_Clean'] = df['Price (English)'].apply(parse_price)

# Convert 'Carpet Area' to numeric, coercing errors to NaN (e.g., 'NA')
df['Carpet Area'] = pd.to_numeric(df['Carpet Area'], errors='coerce')

# Drop rows where 'Price_Clean' or 'Carpet Area' is NaN
df_clean = df.dropna(subset=['Price_Clean', 'Carpet Area'])

# Optionally limit data size to 500 rows for testing
df_sample = df_clean.head(500)

print(f"Data after cleaning: {len(df_sample)} rows")

# Define filter ranges with explicit tuple
price_range = (df_sample['Price_Clean'].min(), df_sample['Price_Clean'].max())
area_range = (df_sample['Carpet Area'].min(), df_sample['Carpet Area'].max())

filtered_df = df_sample[
    (df_sample['Price_Clean'] >= price_range) &
    (df_sample['Price_Clean'] <= price_range[1]) &
    (df_sample['Carpet Area'] >= area_range[0]) &
    (df_sample['Carpet Area'] <= area_range[1])
]

print(f"Rows after filtering: {len(filtered_df)}")
print(filtered_df.head())
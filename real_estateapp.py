import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Load and clean data
@st.cache
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    def parse_price_eng(val):
        if isinstance(val, str):
            val = val.replace(',', '').strip()
            if val.upper() == "NA" or val == "":
                return np.nan
            elif "Lac" in val:
                try: return float(val.replace("Lac", "").strip()) * 100000
                except: return np.nan
            elif "Cr" in val:
                try: return float(val.replace("Cr", "").strip()) * 10000000
                except: return np.nan
            else:
                try: return float(val)
                except: return np.nan
        elif pd.isna(val):
            return np.nan
        else:
            return val

    def clean_carpet_area(val):
        if isinstance(val, str):
            val = val.strip()
            if val.upper() == "NA" or val == "":
                return np.nan
            if "-" in val:
                parts = val.split("-")
                try:
                    low = float(parts[0].strip())
                    high = float(parts[6].strip())
                    return (low + high) / 2
                except:
                    return np.nan
            else:
                try:
                    return float(val)
                except:
                    return np.nan
        elif pd.isna(val):
            return np.nan
        else:
            return val

    df['Price_Clean'] = df['Price (English)'].apply(parse_price_eng)
    df['Carpet Area Clean'] = df['Carpet Area'].apply(clean_carpet_area)
    df_clean = df.dropna(subset=['Price_Clean', 'Carpet Area Clean'])
    return df_clean

st.title("Real Estate KMeans Clustering")

# File upload or use default path
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="kmeans_uploader")


if uploaded_file:
    df_clean = load_and_clean_data(uploaded_file)
else:
    df_clean = load_and_clean_data(r'D:\raj\AIML\real_estate\properties.csv')
    


st.write(f"Dataset loaded. Number of cleaned rows: {len(df_clean)}")

# Select number of clusters
n_clusters = st.slider("Select number of clusters for KMeans", 2, 10, 3)

# Features for clustering
X = df_clean[['Price_Clean', 'Carpet Area Clean']]

# Fit KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)
df_clean['Cluster'] = clusters

# Show cluster centers
st.write("Cluster Centers (Price, Carpet Area):")
st.write(pd.DataFrame(kmeans.cluster_centers_, columns=['Price', 'Carpet Area']))

# Scatter plot colored by cluster
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(df_clean['Carpet Area Clean'], df_clean['Price_Clean'], c=df_clean['Cluster'], cmap='tab10')
plt.colorbar(scatter, ax=ax, label='Cluster')
plt.xlabel('Carpet Area (sqft)')
plt.ylabel('Price (INR)')
plt.title('KMeans Clustering')
st.pyplot(fig)



#ML algorithm

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    df.columns = df.columns.str.strip()
    
    def parse_price_eng(val):
        if isinstance(val, str):
            val = val.replace(',', '').strip()
            if val.upper() == "NA" or val == "":
                return np.nan
            elif "Lac" in val:
                try: return float(val.replace("Lac", "").strip()) * 100000
                except: return np.nan
            elif "Cr" in val:
                try: return float(val.replace("Cr", "").strip()) * 10000000
                except: return np.nan
            else:
                try: return float(val)
                except: return np.nan
        elif pd.isna(val):
            return np.nan
        else:
            return val

    def clean_carpet_area(val):
        if isinstance(val, str):
            val = val.strip()
            if val.upper() == "NA" or val == "":
                return np.nan
            if "-" in val:
                parts = val.split("-")
                try:
                    low = float(parts[0].strip())
                    high = float(parts[10].strip())
                    return (low + high) / 2
                except:
                    return np.nan
            else:
                try:
                    return float(val)
                except:
                    return np.nan
        elif pd.isna(val):
            return np.nan
        else:
            return val

    df['Price_Clean'] = df['Price (English)'].apply(parse_price_eng)
    df['Carpet Area Clean'] = df['Carpet Area'].apply(clean_carpet_area)
    df_clean = df.dropna(subset=['Price_Clean', 'Carpet Area Clean'])
    return df_clean
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="ml_uploader")

st.title("Real Estate ML Modeling")


if uploaded_file:
    df_clean = load_and_clean_data(uploaded_file)
else:
    df_clean = load_and_clean_data(r"D:\raj\AIML\real_estate\properties.csv")

st.write(f"Dataset loaded. Rows after cleaning: {len(df_clean)}")

X = df_clean[['Carpet Area Clean']]
y = df_clean['Price_Clean']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("Model Performance Metrics:")
st.write(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")

st.write("Sample Predictions")
st.write(pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred}).head())







    




























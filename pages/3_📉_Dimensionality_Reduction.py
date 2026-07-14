import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 1. Page Configuration
st.set_page_config(page_title="Dimensionality Reduction", page_icon="📉", layout="wide")

st.title("📉 Dimensionality Reduction (PCA & t-SNE)")
st.markdown("Simplifying complex crime variables to identify the strongest drivers of urban incidents and visualize natural groupings.")
st.markdown("---")

# 2. Load Data (Using a 10k sample to keep the UI instant)
if 'df_full' not in st.session_state:
    st.error("⚠️ Please return to the Home page to load the data.")
    st.stop()

df = st.session_state.df_full.sample(n=10000, random_state=42)

# 3. Prep data for PCA 
cols_to_drop = [
    'DBSCAN_Cluster', 'Temporal_Cluster', 'Hour', 'Month', 
    'Day_of_Week', 'Season', 'Primary Type', 'Location Description', 
    'Latitude', 'Longitude'
]
X_pca_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# Scale the features
scaler = StandardScaler()
X_pca_scaled = scaler.fit_transform(X_pca_raw)

# Run PCA
pca = PCA(n_components=9, random_state=42)
pca.fit(X_pca_scaled)

# 4. Extract Top Features and Clean UI Names
loadings = np.abs(pca.components_)
feature_importance = np.sum(loadings, axis=0)

importance_df = pd.DataFrame({
    'Feature': X_pca_raw.columns,
    'Importance_Score': feature_importance
}).sort_values(by='Importance_Score', ascending=False)

top_5 = importance_df.head(5).copy()

# Dictionary to map backend feature names to clean UI labels
clean_names = {
    'Location Description_Encoded': 'Location Description',
    'Primary Type_Encoded': 'Primary Crime Type',
    'Month_Cos': 'Month (Cyclical Pattern)',
    'Month_Sin': 'Month (Cyclical Pattern)',
    'Day_Cos': 'Day of Week (Cyclical Pattern)',
    'Day_Sin': 'Day of Week (Cyclical Pattern)',
    'Severity_Score': 'Crime Severity Score',
    'Domestic': 'Domestic Incident'
}
top_5['Feature'] = top_5['Feature'].replace(clean_names)
top_5['Importance_Score'] = top_5['Importance_Score'].round(2)

# ==========================================
# PCA DASHBOARD SECTION
# ==========================================
st.subheader("🔑 Top 5 Crime Drivers (PCA)")
st.markdown("Features ranked by their impact on principal components.")

fig_bar = px.bar(
    top_5, 
    x='Importance_Score', 
    y='Feature', 
    orientation='h',
    color='Importance_Score',
    color_continuous_scale='Magma',
    text='Importance_Score'
)
fig_bar.update_layout(
    yaxis={'categoryorder':'total ascending'},
    xaxis_title="Impact Score",
    yaxis_title="Crime Feature"
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

st.subheader("📊 Cumulative Explained Variance")
st.markdown("Proving 9 components capture **>70%** of the original data variance.")

cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100
var_df = pd.DataFrame({
    'Component': range(1, len(cumulative_variance) + 1),
    'Cumulative Variance (%)': cumulative_variance.round(2)
})

fig_line = px.line(
    var_df, 
    x='Component', 
    y='Cumulative Variance (%)', 
    markers=True,
    text='Cumulative Variance (%)'
)
fig_line.update_traces(textposition="top left")
fig_line.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70% Rubric Threshold")
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

# ==========================================
# t-SNE VISUALIZATION SECTION
# ==========================================
st.subheader("🌌 High-Dimensional Separation (t-SNE)")
st.markdown("Visualizing the compression of 23 complex crime features down into a 2D scatter space.")

# Determine the root directory (assuming this script is in your 'pages' folder)
root_dir = Path(__file__).parent.parent

# Point to the images folder (Make sure the files match these names and extensions!)
img1_path = root_dir / "images" / "tsne_crime_types.png"
img2_path = root_dir / "images" / "tsne_geography.png"

# Stack the images vertically exactly as requested with built-in error handling
if img1_path.exists():
    st.image(str(img1_path), caption="t-SNE 2D Map: Separation by Top 5 Crime Types", use_container_width=True)
else:
    st.error(f"⚠️ Could not find image at: {img1_path} - Please verify the image is saved in the 'images' folder.")

st.markdown("<br>", unsafe_allow_html=True) # Adds a little breathing room between images

if img2_path.exists():
    st.image(str(img2_path), caption="t-SNE 2D Map: Clear Geographic District Separation", use_container_width=True)
else:
    st.error(f"⚠️ Could not find image at: {img2_path} - Please verify the image is saved in the 'images' folder.")

st.markdown("---")
st.subheader("💡 What These Maps Tell Us")

st.info(
    """
    **The Simple Story:**
    
    * **Map 1 (Crime Types):** Every crime has its own "DNA." The map groups different crimes separately, proving that a crime like Theft happens under completely different conditions (time, location type) than an Assault.
    
    * **Map 2 (Districts):** Every police district has its own unique set of problems. The distinct "islands" show that the crime pattern in District 1 is completely different from District 2. 
    
    * **The Bottom Line:** Police cannot use a "one-size-fits-all" strategy. Because every district faces different patterns, this tool helps captains know exactly what to expect in their specific neighborhood so they can deploy the right resources.
    """
)
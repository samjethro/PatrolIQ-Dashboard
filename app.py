import streamlit as st
import pandas as pd

# 1. Global Page Configuration
st.set_page_config(
    page_title="PatrolIQ | Safety Analytics",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Lightning-Fast Cached Data Loading
@st.cache_data
def load_full_data():
    """Loads the full 471k dataset for Temporal and PCA analysis."""
    return pd.read_parquet("streamlit_full_data.parquet")

@st.cache_data
def load_geo_sample():
    """Loads the 100k HDBSCAN sample for Geographic Analysis."""
    return pd.read_parquet("streamlit_geo_sample.parquet")

# 3. Load data into session state to make it accessible across pages
if 'df_full' not in st.session_state:
    st.session_state.df_full = load_full_data()

if 'df_geo' not in st.session_state:
    st.session_state.df_geo = load_geo_sample()

# 4. Landing Page UI
st.title("🚨 PatrolIQ - Smart Safety Analytics Platform")
st.markdown("---")

st.markdown("""
### Welcome to the Chicago Crime Intelligence Dashboard

This platform leverages unsupervised machine learning techniques to analyze crime patterns and optimize police resource allocation.

**Platform Capabilities:**
* 🌍 **Geographic Analysis:** Identify distinct high-density crime hotspots using Macro-HDBSCAN.
* 🕒 **Temporal Patterns:** Discover high-risk time slots and seasonal trends using Cyclical K-Means.
* 📉 **Dimensionality Reduction:** Simplify complex crime variables into 2D visualizations using PCA and t-SNE.

👈 **Select an analytics module from the sidebar to begin.**
""")

# Quick dataset overview for the landing page
st.info(f"**Live Database:** Currently analyzing **{len(st.session_state.df_full):,}** recent crime records from Chicago.")
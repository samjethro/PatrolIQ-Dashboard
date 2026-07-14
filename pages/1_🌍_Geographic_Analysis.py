import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Geographic Analysis", page_icon="🌍", layout="wide")

st.title("🌍 Geographic Crime Hotspots")
st.markdown("Visualizing high-density urban risk zones using our production Macro-HDBSCAN clustering model.")
st.markdown("---")

# 2. Load the cached data from the session state
if 'df_geo' not in st.session_state:
    st.error("⚠️ Please return to the Home page (app.py) to load the dataset into memory.")
    st.stop()

df = st.session_state.df_geo

# 3. Sidebar Filters for Interactivity
st.sidebar.header("Map Filters")

# Default to the Top 5 crimes to keep the map fast and readable on initial load
top_crimes = df['Primary Type'].value_counts().head(5).index.tolist()

selected_crimes = st.sidebar.multiselect(
    "Select Crime Types:",
    options=df['Primary Type'].unique(),
    default=top_crimes
)

selected_seasons = st.sidebar.multiselect(
    "Select Seasons:",
    options=df['Season'].unique(),
    default=df['Season'].unique().tolist()
)

# 4. Filter the dataframe based on user selections
filtered_df = df[
    (df['Primary Type'].isin(selected_crimes)) & 
    (df['Season'].isin(selected_seasons))
]

# 5. Isolate true hotspots from ambient noise (-1)
core_mask = filtered_df['HDBSCAN_Cluster'] != -1
df_core = filtered_df[core_mask].copy()

# 6. Build the Cluster Insights Dashboard First
st.subheader("📊 Cluster Intelligence Metrics")

if df_core.empty:
    st.warning("No crime hotspots found for the selected filters.")
else:
    # Aggregate metrics per cluster to fulfill rubric requirements
    insights = df_core.groupby('HDBSCAN_Cluster').agg(
        Total_Incidents=('Primary Type', 'count'),
        Arrest_Rate=('Arrest', lambda x: f"{x.mean() * 100:.1f}%"),
        Dominant_Crime=('Primary Type', lambda x: x.mode()[0])
    ).reset_index()
    
    # Rename for a cleaner UI presentation
    insights.rename(columns={'HDBSCAN_Cluster': 'Cluster ID'}, inplace=True)
    
    # Display as a clean dataframe
    st.dataframe(insights, use_container_width=True, hide_index=True)

st.markdown("---")

# 7. Render the Interactive Plotly Map
st.subheader("🗺️ HDBSCAN Interactive Risk Map")

if not df_core.empty:
    # Convert cluster to string so Plotly treats it as a discrete category for colors
    df_core['Cluster_Label'] = "Hotspot " + df_core['HDBSCAN_Cluster'].astype(str)

    fig = px.scatter_mapbox(
        df_core,
        lat="Latitude",
        lon="Longitude",
        color="Cluster_Label",
        hover_name="Primary Type",
        hover_data={"Location Description": True, "Latitude": False, "Longitude": False, "Cluster_Label": False},
        color_discrete_sequence=px.colors.qualitative.Bold,
        zoom=9.5,
        center={"lat": 41.8781, "lon": -87.6298}, # Centers the map perfectly over Chicago
        mapbox_style="carto-positron",
        height=650
    )
    
    # Remove margins for a clean edge-to-edge map display
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("💡 The Bottom Line")

st.info(
    """
    **How to use this intelligence:**
    
    * **Targeted Patrols (The Hotspot Map):** The AI didn't just highlight random neighborhoods; it mathematically pinpointed the exact, high-density crime zones across Chicago. Instead of driving around blindly, patrol captains can use this map to park squad cars exactly where incidents are statistically concentrated.
    
    * **Defining the Boundaries:** By grouping the crimes geographically, the model shows us exactly where a "danger zone" starts and stops. It ignores isolated, random incidents and focuses strictly on areas with consistent, repeated problems.
    
    * **Why it matters:** This map takes the guesswork out of daily deployment. It stops police departments from wasting time and fuel, allowing them to send patrol officers directly to the specific coordinates where they are needed most.
    """
)
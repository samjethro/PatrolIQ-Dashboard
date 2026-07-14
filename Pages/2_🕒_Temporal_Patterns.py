import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Temporal Patterns", page_icon="🕒", layout="wide")

st.title("🕒 Temporal Crime Patterns")
st.markdown("Analyzing high-risk time slots, seasonal trends, and cyclical crime profiles using K-Means temporal clustering.")
st.markdown("---")

# 2. Load Cached Data
if 'df_full' not in st.session_state:
    st.error("⚠️ Please return to the Home page (app.py) to load the dataset into memory.")
    st.stop()

df = st.session_state.df_full

# 3. Temporal Crime Profiles (Fixed to show Peak/Mode instead of Mean)
st.subheader("📋 K-Means Temporal Crime Profiles")
st.markdown("Peak (most frequent) time-based characteristics for our 4 distinct temporal clusters.")

# FIX: Using .mode() to find the peak hour/day instead of the average to avoid the midnight wrap-around trap
cluster_profiles = df.groupby('Temporal_Cluster')[['Hour', 'Day_of_Week', 'Season']].agg(lambda x: x.mode()[0])
cluster_profiles.index = ["Cluster " + str(i) for i in cluster_profiles.index]
st.dataframe(cluster_profiles, use_container_width=True)

st.markdown("---")

# 4. Vertical Layout for the Interactive Visuals (Removed Columns)
st.subheader("📈 High-Risk Time Slots")
st.markdown("Distribution of crime volume throughout the day by temporal cluster.")

# Aggregate data for the line chart
hourly_cluster_counts = df.groupby(['Hour', 'Temporal_Cluster']).size().reset_index(name='Crime Volume')
hourly_cluster_counts['Temporal_Cluster'] = "Cluster " + hourly_cluster_counts['Temporal_Cluster'].astype(str)

fig_line = px.line(
    hourly_cluster_counts, 
    x='Hour', 
    y='Crime Volume', 
    color='Temporal_Cluster',
    markers=True,
    color_discrete_sequence=px.colors.qualitative.Set1
)

fig_line.update_layout(
    xaxis=dict(tickmode='linear', tick0=0, dtick=2),
    hovermode="x unified",
    legend_title="Temporal Profiles"
)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")

st.subheader("🔥 Hourly Crime Heatmap")
st.markdown("Identifying peak danger zones: Day of the Week vs. Hour of the Day.")

# Matrix for the Heatmap: Rows = Day, Columns = Hour, Values = Count
heatmap_data = df.groupby(['Day_of_Week', 'Hour']).size().unstack(fill_value=0)

# Reorder the days logically instead of alphabetically
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(days_order)

fig_heat = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale='YlOrRd',
    hovertemplate='Hour: %{x}<br>Day: %{y}<br>Crimes: %{z}<extra></extra>'
))

fig_heat.update_layout(
    xaxis=dict(tickmode='linear', tick0=0, dtick=2, title="Hour of the Day (0-23)"),
    yaxis=dict(title="Day of the Week")
)
st.plotly_chart(fig_heat, use_container_width=True)


st.markdown("---")
st.subheader("💡 The Bottom Line")

st.info(
    """
    **How to use this time-based intelligence:**
    
    * **The Seasonal Shift (Bar Charts):** The AI perfectly grouped the data by season entirely on its own. This proves that weather and time of year drastically change crime volumes. What works in the Summer will not work in the Winter.
    
    * **The Daily Rhythm (Hourly Chart):** Crime is not random throughout the day. Notice the massive drop in incidents around 4:00 AM to 5:00 AM, followed by huge, consistent spikes starting at noon and peaking in the evening. 
    
    * **Why it matters:** This dashboard solves the problem of "Police Shift Scheduling." Instead of having the same number of officers on duty at 4 AM as there are at 4 PM, police captains can look at these exact peaks to schedule shift changes and put maximum officers on the street precisely when crime spikes.
    """
)
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import streamlit as st
import pandas as pd
import mlflow
import plotly.express as px



import streamlit as st
import pandas as pd
import mlflow
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Model Monitoring", page_icon="📊", layout="wide")

st.title("📊 Model Performance & MLflow Registry")
st.markdown("Live monitoring of our unsupervised machine learning models, hyperparameters, and evaluation metrics.")
st.markdown("---")

# 2. Connect to MLflow
# This matches the exact setup from your notebook
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db" 
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    # Fetch all experiments
    experiments = mlflow.search_experiments()
    exp_names = [exp.name for exp in experiments]
    
    if not exp_names:
        st.warning("⚠️ Connected to MLflow, but no experiments were found. Ensure your Tracking URI is correct.")
    else:
        # 3. Sidebar Filter for Experiments
        st.sidebar.header("Filter Registry")
        selected_exp = st.sidebar.selectbox("Select Experiment to View:", exp_names)
        
        # Get the ID for the selected experiment
        exp_id = next(exp.experiment_id for exp in experiments if exp.name == selected_exp)
        
        # Fetch runs for this experiment
        runs = mlflow.search_runs(experiment_ids=[exp_id])
        
        if runs.empty:
            st.info(f"No runs logged yet for experiment: {selected_exp}")
        else:
            # 4. Display High-Level Metrics
            st.subheader(f"Experiment Overview: {selected_exp}")
            
            # Clean up the dataframe for display (removing 'tags.', 'metrics.', and 'params.' prefixes)
            clean_runs = runs.copy()
            clean_runs.columns = [col.split('.')[-1] if '.' in col else col for col in clean_runs.columns]
            
            # Display the raw MLflow data cleanly
            st.dataframe(
                clean_runs, 
                use_container_width=True, 
                hide_index=True
            )
            
            st.markdown("---")
            
            # 5. Visualizing Model Performance 
            st.subheader("📈 Performance Metrics Comparison")
            
            # If the user selects the clustering models, show Silhouette Score
            if 'silhouette_score' in clean_runs.columns:
                st.markdown("**Silhouette Score by Run (Target > 0.5)**")
                
                # Use runName if available for better labeling, otherwise default to run_id
                x_axis = 'runName' if 'runName' in clean_runs.columns else 'run_id'
                
                fig = px.bar(
                    clean_runs, 
                    x=x_axis, 
                    y='silhouette_score', 
                    color='silhouette_score',
                    color_continuous_scale='Viridis',
                    text_auto='.3f'
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="0.5 Threshold")
                st.plotly_chart(fig, use_container_width=True)
                
            # If the user selects the PCA model, show Explained Variance Total
            elif 'explained_variance_total' in clean_runs.columns:
                st.markdown("**Explained Variance by Model**")
                
                x_axis = 'runName' if 'runName' in clean_runs.columns else 'run_id'
                
                fig = px.bar(
                    clean_runs, 
                    x=x_axis, 
                    y='explained_variance_total', 
                    color='explained_variance_total',
                    color_continuous_scale='Blues',
                    text_auto='.2f'
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="70% Threshold")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select an experiment with evaluation metrics (like silhouette_score) to see visual comparisons.")

except Exception as e:
    st.error(f"🚨 Failed to connect to MLflow backend. Error: {e}")
    st.info("Make sure your MLFLOW_TRACKING_URI points to the correct location (e.g., your local mlruns folder or SQLite DB).")


# ==========================================
# THE BOTTOM LINE INSIGHTS
# ==========================================
st.markdown("---")
st.subheader("💡 The Bottom Line")

st.info(
    """
    **Why do we track these models?**
    
    * **Accountability:** We aren't just guessing that our algorithms work. MLflow acts as a permanent ledger, proving to stakeholders exactly which hyperparameters (like K-values) produced the best results.
    
    * **Quality Control:** By tracking metrics like the *Silhouette Score* (how well-defined our crime hotspots are) and *Explained Variance*, we ensure our deployed models pass strict mathematical thresholds before they ever reach police captains.
    
    * **Future-Proofing:** If crime patterns shift next year, data scientists can instantly look back at this registry, compare old models to new ones, and upgrade the intelligence platform without starting from scratch.
    """
)
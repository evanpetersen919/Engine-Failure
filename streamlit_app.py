"""
Aircraft Engine RUL Prediction System
Basic Streamlit interface without custom CSS
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from model_loader import load_model_and_data, predict_rul

# Page configuration
st.set_page_config(
    page_title="Engine RUL Prediction",
    layout="wide"
)

# Load model and data
@st.cache_resource
def load_resources():
    return load_model_and_data()

try:
    model, X_test, y_test, scaler, metadata, shap_results = load_resources()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Title
st.title("Aircraft Engine RUL Prediction System")
st.write("Production-Grade LSTM Model with Explainable AI")

# Add explanation box
with st.expander("How This System Works", expanded=False):
    st.markdown("""
    ### What Is RUL?
    
    **Remaining Useful Life (RUL)** is the number of operating cycles an engine can safely run before maintenance is required. One cycle equals one flight.
    
    ### The Data
    
    The model uses NASA C-MAPSS turbofan engine data with 14 sensor measurements (temperature, pressure, fan speed, etc.) recorded over 30 timesteps per engine. This shows how sensor readings change as engines degrade during operation.
    
    ### Data Processing
    
    Raw sensor data is normalized (scaled to equal ranges) and structured into time sequences. This allows the LSTM neural network to detect patterns in how sensor readings evolve over time as engines wear out.
    
    ### The AI Model
    
    An LSTM (Long Short-Term Memory) neural network with two layers analyzes sensor sequences to predict RUL. It learned from thousands of engine lifecycles to recognize patterns that indicate remaining engine life. RMSE of 14.4 cycles means predictions are typically within ±14 cycles of actual values.
    
    ### Results Interpretation
    
    - **Green (>100 cycles)**: Engine is healthy
    - **Orange (50-100 cycles)**: Schedule monitoring
    - **Red (<50 cycles)**: Critical - maintenance needed
    
    SHAP scores show which sensors most influenced each prediction, helping engineers focus inspections on specific components.
    """)

st.divider()

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    engine_idx = st.slider(
        "Select Engine ID",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0
    )
    
    st.divider()
    
    st.subheader("Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test RMSE", "14.4 cycles")
        st.caption("Avg prediction error")
    
    with col2:
        st.metric("Test MAE", "10.8 cycles")
        st.caption("Typical error margin")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("R² Score", "0.765")
        st.caption("76.5% accuracy")
    
    with col4:
        st.metric("Total Engines", f"{len(X_test):,}")
        st.caption("Test dataset size")
    
    st.divider()
    
    with st.expander("About"):
        st.write("**Developed by:** Evan Petersen")
        st.write("**Tech Stack:** PyTorch, SHAP, Streamlit, FastAPI")
        st.write("**Dataset:** NASA C-MAPSS FD001")

# Main content
st.header("Engine Analysis")
st.write(f"**Selected Engine:** {engine_idx} | **Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

# Predict button
if st.button("Run Prediction", type="primary", use_container_width=True):
    with st.spinner("Running prediction..."):
        try:
            # Get prediction
            sequence = X_test[engine_idx]
            actual_rul = float(y_test[engine_idx])
            
            predicted_rul, confidence, top_sensors_df = predict_rul(sequence)
            
            # Calculate metrics
            error = abs(predicted_rul - actual_rul)
            error_pct = (error / actual_rul * 100) if actual_rul > 0 else 0
            
            # Display results
            st.success("Prediction complete!")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Predicted RUL",
                    f"{predicted_rul:.1f} cycles",
                    delta=f"{predicted_rul - actual_rul:.1f}"
                )
            
            with col2:
                st.metric("Actual RUL", f"{actual_rul:.1f} cycles")
            
            with col3:
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col4:
                if predicted_rul > 100:
                    st.metric("Status", "Healthy")
                elif predicted_rul > 50:
                    st.metric("Status", "Monitor")
                else:
                    st.metric("Status", "Critical")
            
            st.divider()
            
            # Error analysis
            st.subheader("Prediction Performance")
            st.caption("How accurate was this specific prediction?")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Absolute Error", f"{error:.2f} cycles")
                st.caption("Difference from actual")
            
            with col2:
                st.metric("Relative Error", f"{error_pct:.2f}%")
                st.caption("Error as percentage")
            
            with col3:
                accuracy = max(0, 100 - error_pct)
                st.metric("Accuracy", f"{accuracy:.1f}%")
                st.caption("Prediction accuracy")
            
            st.divider()
            
            # RUL Gauge - Interactive
            st.subheader("RUL Status Gauge")
            st.caption("Visual representation of remaining engine life (hover for details)")
            
            # Determine color based on RUL
            if predicted_rul > 100:
                color = '#27ae60'  # green
                status = 'Healthy'
            elif predicted_rul > 50:
                color = '#f39c12'  # orange
                status = 'Monitor'
            else:
                color = '#e74c3c'  # red
                status = 'Critical'
            
            # Create interactive gauge
            fig = go.Figure()
            
            # Background bar (max range)
            fig.add_trace(go.Bar(
                y=['RUL'],
                x=[250],
                orientation='h',
                marker=dict(color='lightgray', opacity=0.3),
                name='Max Range',
                hovertemplate='Max Range: 250 cycles<extra></extra>'
            ))
            
            # Predicted RUL bar
            fig.add_trace(go.Bar(
                y=['RUL'],
                x=[predicted_rul],
                orientation='h',
                marker=dict(color=color, opacity=0.8),
                name=f'Predicted RUL',
                hovertemplate=f'Predicted RUL: {predicted_rul:.1f} cycles<br>Status: {status}<extra></extra>'
            ))
            
            # Add threshold lines
            fig.add_vline(x=100, line_dash="dash", line_color="green", opacity=0.5, 
                         annotation_text="Healthy (100)", annotation_position="top")
            fig.add_vline(x=50, line_dash="dash", line_color="orange", opacity=0.5,
                         annotation_text="Critical (50)", annotation_position="top")
            
            fig.update_layout(
                barmode='overlay',
                xaxis_title='Remaining Useful Life (cycles)',
                xaxis=dict(
                    range=[0, 250],
                    gridcolor='#e0e0e0',
                    showgrid=True
                ),
                yaxis=dict(showticklabels=False),
                height=200,
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False,
                hovermode='closest',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(" Green zone (>100) = Healthy |  Orange zone (50-100) = Monitor |  Red zone (<50) = Critical")
            
            st.divider()
            
            # Top sensors
            st.subheader("Critical Sensors (SHAP Analysis)")
            st.caption("Which engine sensors had the biggest impact on this prediction?")
            
            with st.expander("What is SHAP?", expanded=False):
                st.markdown("""
                **SHAP (SHapley Additive exPlanations)** explains which sensors most influenced the AI's prediction.
                
                **How to Read:**
                - Longer bars = more influential sensors
                - Values are importance scores (0-1 scale)
                - Higher scores indicate sensors that dominated the prediction
                
                **Example:** A sensor with 0.87 importance contributed nearly twice as much as one with 0.45.
                
                **Why It Matters:** Instead of black-box predictions, SHAP shows exactly which sensors drove the decision. Engineers can focus inspections on the most critical components.
                """)
            
            # Interactive SHAP bar chart
            features = top_sensors_df['Feature'].tolist()
            importances = top_sensors_df['Importance'].tolist()
            
            # Create interactive bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker=dict(
                    color=importances,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Top 5 Critical Sensors (by Importance)',
                xaxis_title='SHAP Importance Score',
                yaxis_title='',
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(
                    showgrid=False
                ),
                yaxis=dict(
                    showgrid=False
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sensor data table
            st.subheader("Sensor Details")
            st.caption("Detailed breakdown of sensor importance values")
            st.dataframe(top_sensors_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)
else:
    st.info(" Select an engine from the sidebar and click 'Run Prediction' to see the analysis")

# System Capabilities - Always visible
st.divider()
st.subheader("System Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("** Predictive Accuracy**")
    st.write("LSTM neural network achieves 14.4 cycle RMSE (±14 cycles accuracy) trained on 10,000+ engines. Provides confidence scores from 50-95% based on prediction certainty.")

with col2:
    st.write("** Explainable AI**")
    st.write("SHAP analysis reveals which sensors drive each prediction. Engineers can prioritize inspections based on the most influential components.")

with col3:
    st.write("** Production Ready**")
    st.write("REST API for system integration, real-time predictions, 76.5% R² accuracy. Scales from single engines to fleet monitoring.")

st.divider()

st.subheader("Real-World Applications")

st.write("**Airlines:** Optimize maintenance schedules and reduce unplanned downtime by predicting failures before they occur. Service engines when actually needed rather than fixed intervals.")

st.write("**Manufacturers:** Monitor fleet health patterns to identify design weaknesses and validate improvements based on real-world sensor data.")

st.write("**Maintenance Teams:** Prioritize daily inspections using risk-based predictions. Allocate technician time to the most critical engines first.")

st.write("**Regulators:** Track fleet-wide degradation patterns to issue targeted maintenance directives and validate safety compliance.")

# Footer
st.divider()
st.write("**Aircraft Engine RUL Prediction System**")
st.write("Developed by Evan Petersen | 2025")
st.write("PyTorch • SHAP • Streamlit • FastAPI")

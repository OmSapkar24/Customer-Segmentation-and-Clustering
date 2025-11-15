import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.segmenter import Segmenter
from src.preprocessing import preprocess_data
from src.visualization import plot_clusters_2d, plot_clusters_3d
from src.metrics import evaluate_clustering
import io

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("👥 Customer Segmentation Dashboard")
st.markdown("""
    Discover actionable customer segments using advanced clustering algorithms.
    Upload your data, select parameters, and get instant insights!
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Customer Data (CSV)",
        type=["csv"],
        help="Upload a CSV file containing customer data"
    )
    
    st.markdown("---")
    
    # Algorithm selection
    algorithm = st.selectbox(
        "Clustering Algorithm",
        ["K-Means", "DBSCAN", "GMM (Gaussian Mixture)"],
        help="Choose the clustering algorithm to use"
    )
    
    # Algorithm-specific parameters
    if algorithm == "K-Means" or algorithm == "GMM (Gaussian Mixture)":
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=4,
            help="Select the number of customer segments"
        )
    else:  # DBSCAN
        eps = st.slider(
            "Epsilon (neighborhood size)",
            min_value=0.1,
            max_value=2.0,
            value=0.5,
            step=0.1
        )
        min_samples = st.slider(
            "Minimum Samples",
            min_value=2,
            max_value=20,
            value=5
        )
    
    st.markdown("---")
    
    # Additional options
    st.subheader("📊 Visualization Options")
    show_3d = st.checkbox("Show 3D Visualization", value=True)
    show_metrics = st.checkbox("Show Clustering Metrics", value=True)
    
    # Run button
    run_segmentation = st.button("🚀 Run Segmentation", type="primary")

# Main content area
if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
        
        # Display data preview
        with st.expander("📋 Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
        
        if run_segmentation:
            with st.spinner("🔄 Processing data and generating segments..."):
                # Preprocess data
                X_scaled, scaler = preprocess_data(df)
                
                # Configure segmenter
                method_map = {
                    "K-Means": "kmeans",
                    "DBSCAN": "dbscan",
                    "GMM (Gaussian Mixture)": "gmm"
                }
                method = method_map[algorithm]
                
                # Initialize segmenter with appropriate parameters
                if method == "dbscan":
                    segmenter = Segmenter(method=method, eps=eps, min_samples=min_samples)
                else:
                    segmenter = Segmenter(method=method, n_clusters=n_clusters)
                
                # Fit and predict
                labels = segmenter.fit_predict(X_scaled)
                
                # Add segments to dataframe
                df['Segment'] = labels
                
                # Display results
                st.success("✅ Segmentation completed successfully!")
                
                # Metrics section
                if show_metrics and method != "dbscan":
                    st.subheader("📈 Clustering Metrics")
                    metrics = evaluate_clustering(X_scaled, labels)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Silhouette Score",
                            f"{metrics.get('silhouette', 0):.3f}",
                            help="Measures how similar objects are to their own cluster vs other clusters"
                        )
                    with col2:
                        st.metric(
                            "Davies-Bouldin Index",
                            f"{metrics.get('davies_bouldin', 0):.3f}",
                            help="Lower values indicate better clustering"
                        )
                    with col3:
                        st.metric(
                            "Calinski-Harabasz Score",
                            f"{metrics.get('calinski_harabasz', 0):.1f}",
                            help="Higher values indicate better defined clusters"
                        )
                
                # Segment distribution
                st.subheader("📊 Segment Distribution")
                segment_counts = df['Segment'].value_counts().sort_index()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_bar = px.bar(
                        x=segment_counts.index,
                        y=segment_counts.values,
                        labels={'x': 'Segment', 'y': 'Customer Count'},
                        title='Customers per Segment',
                        color=segment_counts.index,
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    fig_pie = px.pie(
                        values=segment_counts.values,
                        names=segment_counts.index,
                        title='Segment Proportion',
                        hole=0.4
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Segment profiles
                st.subheader("🎯 Segment Profiles")
                profile = segmenter.profile(df.drop('Segment', axis=1), labels)
                st.dataframe(profile, use_container_width=True)
                
                # Visualizations
                st.subheader("🔍 Cluster Visualizations")
                
                # 2D visualization
                st.markdown("#### 2D Projection (PCA)")
                fig_2d = plot_clusters_2d(X_scaled, labels)
                st.plotly_chart(fig_2d, use_container_width=True)
                
                # 3D visualization
                if show_3d:
                    st.markdown("#### 3D Projection (PCA)")
                    fig_3d = plot_clusters_3d(X_scaled, labels)
                    st.plotly_chart(fig_3d, use_container_width=True)
                
                # Download results
                st.subheader("💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download segmented data
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Segmented Data (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="segmented_customers.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Download segment profiles
                    profile_buffer = io.StringIO()
                    profile.to_csv(profile_buffer, index=False)
                    st.download_button(
                        label="📥 Download Segment Profiles (CSV)",
                        data=profile_buffer.getvalue(),
                        file_name="segment_profiles.csv",
                        mime="text/csv"
                    )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format and contains numerical data.")

else:
    # Welcome message when no file is uploaded
    st.info("👈 Please upload a CSV file to get started")
    
    # Show example data format
    with st.expander("📖 Expected Data Format"):
        st.markdown("""
            Your CSV file should contain customer data with numerical features such as:
            - Customer ID
            - Age
            - Annual Income
            - Spending Score
            - Purchase Frequency
            - etc.
            
            **Example:**
        """)
        
        example_df = pd.DataFrame({
            'CustomerID': [1, 2, 3, 4, 5],
            'Age': [25, 35, 42, 28, 51],
            'AnnualIncome': [50000, 70000, 65000, 55000, 90000],
            'SpendingScore': [45, 78, 62, 51, 85],
            'PurchaseFrequency': [3, 8, 5, 4, 10]
        })
        st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit | Customer Segmentation & Clustering for Growth</p>
    </div>
""", unsafe_allow_html=True)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configurazione pagina
st.set_page_config(
    page_title="ChemometricSolutions - Interactive Demos",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1E90FF 0%, #4169E1 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .demo-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: #f8f9fa;
        transition: box-shadow 0.3s ease;
    }
    .demo-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
    }
    .stat-box {
        text-align: center;
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E90FF;
    }
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header principale
st.markdown("""
<div class="main-header">
    <h1>🧪 ChemometricSolutions</h1>
    <h3>Interactive Demos & Tools</h3>
    <p>Transform complex challenges into interpretable solutions with chemometrics</p>
</div>
""", unsafe_allow_html=True)

# Sidebar con info
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1E90FF/ffffff?text=ChemometricSolutions", width=250)
    st.markdown("---")
    st.markdown("### 👥 Our Team")
    st.markdown("**Prof. Riccardo Leardi**  \nUniversity of Genova")
    st.markdown("**Dr. Emanuele Farinini**  \nChemometrics Expert")
    
    st.markdown("---")
    st.markdown("### 📧 Contact")
    st.markdown("info@chemometricsolutions.com")
    st.markdown("[Visit Main Website](https://chemometricsolutions.com)")

# Statistiche principali
st.markdown("""
<div class="stats-container">
    <div class="stat-box">
        <div class="stat-number">35+</div>
        <div class="stat-label">Years Experience</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">100+</div>
        <div class="stat-label">Projects Completed</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">150+</div>
        <div class="stat-label">Publications</div>
    </div>
    <div class="stat-box">
        <div class="stat-number">30+</div>
        <div class="stat-label">Client Companies</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Demo disponibili
st.markdown("## 🚀 Available Demos")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="demo-card">
        <h4>📊 PCA Analysis Tool</h4>
        <p>Interactive Principal Component Analysis with real-time visualization. 
        Upload your data and explore multivariate relationships.</p>
        <ul>
            <li>Scores and loadings plots</li>
            <li>Explained variance analysis</li>
            <li>Data preprocessing options</li>
            <li>Export results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔗 Open PCA Demo", key="pca"):
        st.switch_page("pages/1_📊_PCA_Analysis.py")

with col2:
    st.markdown("""
    <div class="demo-card">
        <h4>🎯 DoE Simulator</h4>
        <p>Design of Experiments planning and simulation tool. 
        Create optimal experimental designs for your research.</p>
        <ul>
            <li>Factorial designs</li>
            <li>Response surface methodology</li>
            <li>Optimization strategies</li>
            <li>Statistical analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔗 Open DoE Demo", key="doe"):
        st.switch_page("pages/2_🎯_DoE_Simulator.py")

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="demo-card">
        <h4>📈 Process Monitor</h4>
        <p>Real-time process monitoring dashboard with control charts 
        and multivariate statistical process control.</p>
        <ul>
            <li>Control charts (Shewhart, CUSUM)</li>
            <li>Multivariate monitoring</li>
            <li>Anomaly detection</li>
            <li>Real-time alerts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔗 Open Process Monitor", key="process"):
        st.switch_page("pages/3_📈_Process_Monitor.py")

with col4:
    st.markdown("""
    <div class="demo-card">
        <h4>🔬 Calibration Tool</h4>
        <p>Multivariate calibration methods including PLS regression 
        for quantitative analysis applications.</p>
        <ul>
            <li>PLS regression</li>
            <li>Cross-validation</li>
            <li>Model diagnostics</li>
            <li>Prediction intervals</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔗 Open Calibration Demo", key="calibration"):
        st.switch_page("pages/4_🔬_Calibration_Tool.py")

# Sezione About
st.markdown("---")
st.markdown("## 🎓 About ChemometricSolutions")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **ChemometricSolutions** provides advanced consulting and training in multivariate 
    statistical analysis for the chemical, pharmaceutical, food, and environmental industries.
    
    Our expertise spans:
    - **Design of Experiments (DoE)** - Optimal experimental planning
    - **Multivariate Analysis** - PCA, PLS, MLR, and advanced methods  
    - **Process Monitoring** - Statistical process control and optimization
    - **Method Development** - Analytical method optimization and validation
    - **Training Programs** - Customized education for teams and individuals
    
    Founded by **Prof. Riccardo Leardi** (University of Genova) with over 35 years 
    of experience in chemometrics and **Dr. Emanuele Farinini**, we combine academic 
    rigor with practical industry experience.
    """)

with col2:
    # Grafico semplice delle applicazioni
    industries = ['Pharmaceutical', 'Food & Beverage', 'Chemical', 'Environmental', 'Other']
    values = [35, 25, 20, 15, 5]
    
    fig = px.pie(
        values=values, 
        names=industries, 
        title="Our Client Industries",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Call to Action
st.markdown("---")
st.markdown("## 💼 Ready to Transform Your Data Analysis?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📚 Training")
    st.markdown("Custom training programs tailored to your team's needs.")
    
with col2:
    st.markdown("### 🔬 Consulting")
    st.markdown("Expert guidance for your chemometric challenges.")
    
with col3:
    st.markdown("### 🛠️ Software")
    st.markdown("Custom applications and tool development.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>ChemometricSolutions</strong> | Advanced Multivariate Statistical Solutions</p>
    <p>📧 info@chemometricsolutions.com | 🌐 <a href="https://chemometricsolutions.com">chemometricsolutions.com</a></p>
    <p><em>Transforming complex challenges into interpretable solutions</em></p>
</div>
""", unsafe_allow_html=True)
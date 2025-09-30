"""
Advanced PCA Diagnostics - UNIFIED VERSION
- Fixed show_sample_names checkbox functionality
- Added dark mode toggle with unified color schemes
- Enhanced visualization options
- Consistent with color_utils.py system
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import chi2, f
from scipy import stats

# Import sistema di colori unificato
from color_utils import get_unified_color_schemes, create_categorical_color_map

def calculate_t2_statistic(scores, n_components=None):
    """Calculate T² (Hotelling's T²) statistic for PCA scores"""
    if n_components is None:
        n_components = scores.shape[1]
    
    scores_subset = scores.iloc[:, :n_components]
    t2_values = np.sum(scores_subset.values**2, axis=1)
    
    return t2_values


def calculate_q_statistic(processed_data, scores, loadings, n_components=None):
    """Calculate Q statistic (SPE - Squared Prediction Error)"""
    if n_components is None:
        n_components = min(scores.shape[1], loadings.shape[1])
    
    scores_subset = scores.iloc[:, :n_components].values
    loadings_subset = loadings.iloc[:, :n_components].values
    
    reconstructed = scores_subset @ loadings_subset.T
    residuals = processed_data.values - reconstructed
    q_values = np.sum(residuals**2, axis=1)
    
    return q_values


def calculate_control_limits(n_samples, n_variables, n_components, confidence_levels=[0.95, 0.99, 0.999]):
    """Calculate control limits for T² and Q statistics"""
    limits = {'t2': {}, 'q': {}}
    
    # T² control limits using F-distribution
    for conf in confidence_levels:
        factor = (n_components * (n_samples**2 - 1)) / (n_samples * (n_samples - n_components))
        f_critical = f.ppf(conf, n_components, n_samples - n_components)
        t2_limit = factor * f_critical
        limits['t2'][conf] = t2_limit
    
    # Q control limits (simplified approach using chi-square)
    for conf in confidence_levels:
        dof = n_variables - n_components
        q_limit = chi2.ppf(conf, dof)
        limits['q'][conf] = q_limit
    
    return limits


def create_t2_q_plot(t2_values, q_values, control_limits, sample_names=None, 
                     title="T² vs Q Plot", color_data=None, color_variable=None, 
                     show_sample_names=True, dark_mode=False):
    """
    Create T² vs Q diagnostic plot with FIXED color consistency
    """
    # Get unified color scheme
    colors = get_unified_color_schemes(dark_mode)
    
    # Handle sample names properly
    if sample_names is None or not show_sample_names:
        display_names = None
        hover_text = [f"Sample_{i+1}" for i in range(len(t2_values))]
    else:
        display_names = sample_names
        hover_text = sample_names
    
    # Create base scatter plot with FORCED color consistency
    if color_data is not None:
        # SEMPRE crea color_discrete_map per garantire coerenza
        unique_values = pd.Series(color_data).dropna().unique()
        color_discrete_map = create_categorical_color_map(unique_values, dark_mode)
        
        fig = px.scatter(
            x=t2_values,
            y=q_values,
            color=color_data,
            text=display_names if show_sample_names else None,
            title=title,
            labels={'x': 'T² Statistic', 'y': 'Q Statistic', 'color': color_variable},
            color_discrete_map=color_discrete_map  # FORZATO sempre
        )
        
        # Update hover template with proper text handling
        if color_variable:
            hover_template = f'<b>%{{hovertext}}</b><br>T²: %{{x:.2f}}<br>Q: %{{y:.2f}}<br>{color_variable}: %{{marker.color}}<extra></extra>'
        else:
            hover_template = '<b>%{hovertext}</b><br>T²: %{x:.2f}<br>Q: %{y:.2f}<extra></extra>'
            
        fig.update_traces(
            hovertemplate=hover_template,
            hovertext=hover_text,
            marker=dict(size=8, opacity=0.7),
            textposition="top center" if show_sample_names else None
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t2_values,
            y=q_values,
            mode='markers+text' if show_sample_names else 'markers',
            name='Samples',
            text=display_names if show_sample_names else None,
            hovertext=hover_text,
            hovertemplate='<b>%{hovertext}</b><br>T²: %{x:.2f}<br>Q: %{y:.2f}<extra></extra>',
            marker=dict(
                size=8,
                color=colors['point_color'],
                opacity=0.7
            ),
            textposition="top center"
        ))
    
    # Add control limit lines with theme colors
    conf_labels = ['95%', '99%', '99.9%']
    
    for i, (conf, color, label) in enumerate(zip([0.95, 0.99, 0.999], colors['control_colors'], conf_labels)):
        if conf in control_limits['t2']:
            # T² vertical line
            fig.add_vline(
                x=control_limits['t2'][conf],
                line=dict(color=color, width=2, dash='dash'),
                annotation_text=f"T² {label}",
                annotation_position="top"
            )
        
        if conf in control_limits['q']:
            # Q horizontal line  
            fig.add_hline(
                y=control_limits['q'][conf],
                line=dict(color=color, width=2, dash='dash'),
                annotation_text=f"Q {label}",
                annotation_position="right"
            )
    
    # Apply unified styling
    fig.update_layout(
        title=title,
        xaxis_title="T² Statistic",
        yaxis_title="Q Statistic", 
        height=600,
        showlegend=True,
        hovermode='closest',
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font=dict(color=colors['text']),
        xaxis=dict(
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid']
        ),
        yaxis=dict(
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid']
        )
    )
    
    return fig

def create_time_series_plots(t2_values, q_values, control_limits, sample_names, 
                           timestamps=None, dark_mode=False):
    """Create T² and Q time series plots with unified color scheme"""
    colors = get_unified_color_schemes(dark_mode)
    
    # Use timestamps if provided, otherwise sample numbers
    if timestamps is not None:
        x_axis = timestamps
        x_title = "Time"
    else:
        x_axis = list(range(1, len(t2_values) + 1))
        x_title = "Sample Number"
    
    # T² time series plot
    t2_fig = go.Figure()
    
    t2_fig.add_trace(go.Scatter(
        x=x_axis,
        y=t2_values,
        mode='lines+markers',
        name='T² Values',
        text=sample_names,
        hovertemplate='<b>%{text}</b><br>' + x_title + ': %{x}<br>T²: %{y:.2f}<extra></extra>',
        line=dict(color=colors['line_colors'][0], width=2),
        marker=dict(size=4)
    ))
    
    # Add T² control limits
    conf_labels = ['95%', '99%', '99.9%']
    
    for conf, color, label in zip([0.95, 0.99, 0.999], colors['control_colors'], conf_labels):
        if conf in control_limits['t2']:
            t2_fig.add_hline(
                y=control_limits['t2'][conf],
                line=dict(color=color, width=2, dash='dash'),
                annotation_text=f"T² {label}",
                annotation_position="right"
            )
    
    t2_fig.update_layout(
        title="T² Statistic Over Time",
        xaxis_title=x_title,
        yaxis_title="T² Statistic",
        height=400,
        showlegend=False,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font=dict(color=colors['text']),
        xaxis=dict(gridcolor=colors['grid']),
        yaxis=dict(gridcolor=colors['grid'])
    )
    
    # Q time series plot
    q_fig = go.Figure()
    
    q_fig.add_trace(go.Scatter(
        x=x_axis,
        y=q_values,
        mode='lines+markers',
        name='Q Values',
        text=sample_names,
        hovertemplate='<b>%{text}</b><br>' + x_title + ': %{x}<br>Q: %{y:.2f}<extra></extra>',
        line=dict(color=colors['line_colors'][1], width=2),
        marker=dict(size=4)
    ))
    
    # Add Q control limits
    for conf, color, label in zip([0.95, 0.99, 0.999], colors['control_colors'], conf_labels):
        if conf in control_limits['q']:
            q_fig.add_hline(
                y=control_limits['q'][conf],
                line=dict(color=color, width=2, dash='dash'),
                annotation_text=f"Q {label}",
                annotation_position="right"
            )
    
    q_fig.update_layout(
        title="Q Statistic Over Time",
        xaxis_title=x_title,
        yaxis_title="Q Statistic",
        height=400,
        showlegend=False,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font=dict(color=colors['text']),
        xaxis=dict(gridcolor=colors['grid']),
        yaxis=dict(gridcolor=colors['grid'])
    )
    
    return t2_fig, q_fig


def show_advanced_diagnostics_tab(processed_data, scores, pca_params, timestamps=None, start_sample=1):
    """
    UNIFIED VERSION - Main function to display advanced diagnostics tab
    """
    
    st.markdown("### 🎯 T² vs Q Diagnostic Analysis")
    st.info("Advanced outlier detection using Hotelling's T² and Q statistics")
    
    # Parameter selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_components = st.slider(
            "Number of components for diagnostics:",
            min_value=1,
            max_value=min(scores.shape[1], 10),
            value=min(3, scores.shape[1]),
            help="More components = more sensitive T², fewer components = more sensitive Q"
        )
    
    with col2:
        diagnostic_type = st.selectbox(
            "Diagnostic approach:",
            ["Independent T²/Q", "Joint T²/Q"],
            help="Independent: separate thresholds | Joint: hierarchical classification"
        )
    
    with col3:
        # FIXED: Properly handle the checkbox state
        show_sample_names = st.checkbox("Show sample names on plots", value=False, key="show_names_diag")
    
    # === UNIFIED VISUALIZATION OPTIONS ===
    st.markdown("### 🎨 Visualization Options")
    
    # Get original dataset for coloring options
    try:
        original_data = st.session_state.current_data
        
        # Get custom variables
        custom_vars = []
        if 'custom_variables' in st.session_state:
            custom_vars = list(st.session_state.custom_variables.keys())
        
        # All available coloring options (exclude variables used in PCA)
        pca_variables = pca_params.get('parameters', {}).get('variables', [])
        available_color_vars = [col for col in original_data.columns if col not in pca_variables]
        
        col_viz1, col_viz2, col_viz3 = st.columns(3)
        
        with col_viz1:
            all_color_options = (["None", "Row Index"] + available_color_vars + custom_vars)
            color_by = st.selectbox("Color points by:", all_color_options, key="diag_color")
        
        with col_viz2:
            plot_type = st.selectbox(
                "Additional plots:",
                ["T² vs Q Only", "Include Time Series"],
                help="Time series shows T² and Q trends over sample number/time"
            )
        
        with col_viz3:
            # Dark mode toggle - consistent with PCA.py pattern
            use_dark_theme = st.checkbox("🌙 Dark mode colors", value=True, key="dark_mode_diag_colors")
        
        # Prepare color data
        color_data = None
        color_variable = None
        
        if color_by != "None":
            color_variable = color_by
            if color_by == "Row Index":
                color_data = [f"Sample_{i+1}" for i in range(len(processed_data))]
            elif color_by in custom_vars:
                color_data = st.session_state.custom_variables[color_by].reindex(processed_data.index).fillna("Unknown")
            else:
                color_data = original_data[color_by].reindex(processed_data.index).fillna("Unknown")
                
    except Exception as e:
        # Fallback if original data not available
        color_data = None
        color_variable = None
        plot_type = "T² vs Q Only"
        use_dark_theme = True
        st.info("Original dataset not available for coloring options")
    
    # Calculate diagnostics
    with st.spinner("Calculating T² and Q statistics..."):
        try:
            # Calculate T² statistic
            t2_values = calculate_t2_statistic(scores, n_components)
            
            # Calculate Q statistic  
            if 'loadings' in pca_params:
                loadings = pca_params['loadings']
                q_values = calculate_q_statistic(processed_data, scores, loadings, n_components)
            else:
                st.error("❌ PCA loadings not found in model parameters")
                return
            
            # Calculate control limits
            n_samples, n_variables = processed_data.shape
            control_limits = calculate_control_limits(n_samples, n_variables, n_components)
            
            # FIXED: Get sample names based on checkbox state
            if show_sample_names:
                sample_names = processed_data.index.tolist()
            else:
                sample_names = [f"Sample_{i+1}" for i in range(len(t2_values))]
            
            st.success("✅ Diagnostics calculated successfully")
            
        except Exception as e:
            st.error(f"❌ Error calculating diagnostics: {str(e)}")
            return
    
    # Display results based on diagnostic type
    if diagnostic_type == "Independent T²/Q":
        st.markdown("### 📊 Independent T² and Q Analysis")
        st.info("Separate analysis of T² and Q statistics with independent thresholds")
        
        # Summary metrics for all three levels
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples Analyzed", len(t2_values))
        
        with col2:
            t2_outliers_95 = sum(t2_values > control_limits['t2'][0.95])
            st.metric("T² Outliers (95%)", t2_outliers_95)
        
        with col3:
            q_outliers_95 = sum(q_values > control_limits['q'][0.95])
            st.metric("Q Outliers (95%)", q_outliers_95)
        
        with col4:
            combined_95 = sum((t2_values > control_limits['t2'][0.95]) | 
                             (q_values > control_limits['q'][0.95]))
            st.metric("Any Outliers (95%)", combined_95)
        
        # Display current settings
        settings_info = f"**Settings**: {n_components} components, "
        if show_sample_names:
            settings_info += "Names ON, "
        else:
            settings_info += "Names OFF, "
        
        if use_dark_theme:
            settings_info += "Dark Mode"
        else:
            settings_info += "Light Mode"
        
        st.info(settings_info)
        
        # Color information display
        if color_data is not None:
            st.info(f"Points colored by: **{color_variable}** ({len(pd.Series(color_data).unique())} categories)")
        
        # UNIFIED: T² vs Q plot with proper sample names handling and unified colors
        fig_diagnostic = create_t2_q_plot(
            t2_values, q_values, control_limits, 
            sample_names, 
            f"Independent T² vs Q Analysis ({n_components} components)",
            color_data=color_data,
            color_variable=color_variable,
            show_sample_names=show_sample_names,
            dark_mode=use_dark_theme
        )
        st.plotly_chart(fig_diagnostic, use_container_width=True)
        
        # Time series plots if requested
        if plot_type == "Include Time Series":
            st.markdown("### 📈 Time Series Analysis")
            
            t2_time_fig, q_time_fig = create_time_series_plots(
                t2_values, q_values, control_limits, sample_names, 
                timestamps, dark_mode=use_dark_theme
            )
            
            col_t2, col_q = st.columns(2)
            with col_t2:
                st.plotly_chart(t2_time_fig, use_container_width=True)
            with col_q:
                st.plotly_chart(q_time_fig, use_container_width=True)
    
    else:  # Joint T²/Q analysis
        st.markdown("### 📊 Joint T² and Q Analysis")
        st.info("Hierarchical classification: samples assigned to highest violated threshold level")
        
        # Implement joint analysis with same unified approach
        st.info("🚧 Joint analysis implementation coming soon with unified colors")
    
    # Status summary
    st.markdown("### 📋 Current Settings Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Components Used", n_components)
    with col2:
        st.metric("Analysis Type", diagnostic_type.split()[0])
    with col3:
        status_names = "✅ ON" if show_sample_names else "❌ OFF"
        st.metric("Sample Names", status_names)
    with col4:
        mode_status = "🌙 Dark" if use_dark_theme else "☀️ Light"
        st.metric("Display Mode", mode_status)
    
    # Interpretation guide with mode-specific styling
    st.markdown("### 📋 Interpretation Guide")
    
    # Use unified color scheme for styling
    colors = get_unified_color_schemes(use_dark_theme)
    
    if use_dark_theme:
        guide_style = """
        <div style='background-color: #2d2d2d; padding: 1rem; border-radius: 5px; color: #ffffff; border-left: 4px solid #00ff00;'>
        """
    else:
        guide_style = """
        <div style='background-color: #f0f8ff; padding: 1rem; border-radius: 5px; border-left: 4px solid green;'>
        """
    
    with st.expander("📖 Understanding the Analysis"):
        st.markdown(guide_style + """
        **Unified Color System Features:**
        - ✅ Consistent colors with main PCA plots
        - ✅ Proper categorical variable coloring
        - ✅ Dark/light mode compatibility
        - ✅ Sample names display control
        
        **Control Lines (Color-coded):**
        - **Green**: 95% confidence limit (routine monitoring)
        - **Orange**: 99% confidence limit (warning level)  
        - **Red**: 99.9% confidence limit (alarm level)
        </div>""", unsafe_allow_html=True)
    
    # Export section
    st.markdown("### 💾 Export Results")
    
    if st.button("📊 Export Analysis Data", type="primary"):
        # Create comprehensive export
        export_df = pd.DataFrame({
            'Sample': sample_names,
            'T2_Statistic': t2_values,
            'Q_Statistic': q_values,
            'Settings': f"{n_components}_components_{diagnostic_type.replace(' ', '_').replace('²', '2')}"
        })
        
        if color_data is not None:
            export_df[color_variable] = color_data
        
        csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            "📥 Download Results CSV",
            csv_data,
            f"PCA_Diagnostics_Unified_{diagnostic_type.replace(' ', '_').replace('²', '2')}.csv",
            "text/csv"
        )
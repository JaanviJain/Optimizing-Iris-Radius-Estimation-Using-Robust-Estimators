import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import HuberRegressor, RANSACRegressor
import os
import warnings
warnings.filterwarnings('ignore')

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# Load your data
df = pd.read_excel(r"D:\SEM-5\ELECTIVE'S\DEPARTMENT-ELECTIVE\BIOMETRICS\Project\iris_measurements.xlsx")

print("Creating interactive visualizations...")

# Define angle sets
angles_6 = [30, 0, -30, 150, 180, 210]
angles_8 = [30, 0, -30, 150, 180, 210, 90, 270]

angle_columns_6 = [f'radius_{angle}' for angle in angles_6]
angle_columns_8 = [f'radius_{angle}' for angle in angles_8]

# ========== FIXED ANALYSIS FUNCTION ==========

def analyze_angle_set_interactive(df, angle_columns, set_name):
    """Comprehensive analysis with detailed metrics for interactive visualization"""
    
    # Apply estimators
    def apply_estimators(row, angles):
        radii = row[angles].values
        results = {}
        
        # Basic estimators
        results['mean'] = np.mean(radii)
        results['median'] = np.median(radii)
        results['trimmed_mean_20'] = stats.trim_mean(radii, proportiontocut=0.2)
        results['midmean'] = np.mean(np.percentile(radii, [25, 75]))
        
        # Advanced estimators
        X_dummy = np.ones((len(radii), 1))
        
        try:
            huber = HuberRegressor().fit(X_dummy, radii)
            results['huber'] = huber.coef_[0]
            results['huber_success'] = 1
        except:
            results['huber'] = np.median(radii)
            results['huber_success'] = 0
        
        try:
            ransac = RANSACRegressor(random_state=42).fit(X_dummy, radii)
            results['ransac'] = ransac.estimator_.coef_[0]
            results['ransac_success'] = 1
        except:
            results['ransac'] = np.median(radii)
            results['ransac_success'] = 0
        
        return pd.Series(results)
    
    # Apply estimators
    estimator_results = df.apply(lambda row: apply_estimators(row, angle_columns), axis=1)
    df_temp = df.copy()
    df_temp = pd.concat([df_temp, estimator_results], axis=1)
    
    # Calculate comprehensive stability metrics
    stability_data = []
    estimators = ['mean', 'median', 'trimmed_mean_20', 'midmean', 'huber', 'ransac']
    
    for (person_id, eye), group in df_temp.groupby(['person_id', 'eye']):
        if len(group) > 1:  # Only calculate stability if we have multiple images
            eye_data = {
                'person_id': person_id,
                'eye': eye,
                'num_images': len(group)
            }
            
            for estimator in estimators:
                radii = group[estimator].values
                
                # Calculate MAD (Median Absolute Deviation)
                mad_value = np.median(np.abs(radii - np.median(radii)))
                
                # Calculate standard deviation, handle case where it might be 0
                std_value = np.std(radii)
                if std_value == 0:
                    std_value = 0.001  # Small value to avoid division issues
                
                eye_data.update({
                    f'{estimator}_std': std_value,
                    f'{estimator}_mad': mad_value,
                    f'{estimator}_iqr': stats.iqr(radii),
                    f'{estimator}_mean': np.mean(radii),
                    f'{estimator}_min': np.min(radii),
                    f'{estimator}_max': np.max(radii),
                    f'{estimator}_range': np.ptp(radii)
                })
            
            stability_data.append(eye_data)
    
    stability_df = pd.DataFrame(stability_data)
    
    # Calculate performance summary
    performance_data = []
    for estimator in estimators:
        std_values = stability_df[f'{estimator}_std'].dropna()
        mad_values = stability_df[f'{estimator}_mad'].dropna()
        iqr_values = stability_df[f'{estimator}_iqr'].dropna()
        
        # Only include if we have valid data
        if len(std_values) > 0:
            performance_data.append({
                'estimator': estimator,
                'set_name': set_name,
                'avg_std': std_values.mean(),
                'avg_mad': mad_values.mean(),
                'avg_iqr': iqr_values.mean(),
                'min_std': std_values.min(),
                'max_std': std_values.max(),
                'std_of_std': std_values.std(),
                'num_eyes': len(std_values)
            })
    
    performance_df = pd.DataFrame(performance_data)
    
    return stability_df, performance_df, df_temp

# ========== VISUALIZATION FUNCTIONS ==========

def create_radar_chart(performance_df, title):
    """Create radar chart for performance comparison"""
    categories = ['Stability (1/Std)', 'Robustness (1/MAD)', 'Consistency (1/IQR)']
    
    fig = go.Figure()
    
    for _, row in performance_df.iterrows():
        # Invert values so higher is better
        stability = 1 / row['avg_std'] if row['avg_std'] > 0 else 0
        robustness = 1 / row['avg_mad'] if row['avg_mad'] > 0 else 0
        consistency = 1 / row['avg_iqr'] if row['avg_iqr'] > 0 else 0
        
        # Normalize to 0-1 scale for radar chart
        max_val = max(stability, robustness, consistency)
        if max_val > 0:
            stability /= max_val
            robustness /= max_val
            consistency /= max_val
        
        fig.add_trace(go.Scatterpolar(
            r=[stability, robustness, consistency, stability],
            theta=categories + [categories[0]],
            fill='toself',
            name=row['estimator_label']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title=title,
        height=600
    )
    
    return fig

# ========== PHASE 1: 6-ANGLES INTERACTIVE VISUALIZATION ==========

print("Creating 6-angle interactive visualizations...")
stability_6, performance_6, df_6 = analyze_angle_set_interactive(df, angle_columns_6, "6 Angles")

# Map estimator names to readable labels
estimator_names = {
    'mean': 'Mean',
    'median': 'Median', 
    'trimmed_mean_20': 'Trimmed Mean (20%)',
    'midmean': 'Midmean',
    'huber': 'Huber',
    'ransac': 'RANSAC'
}

performance_6['estimator_label'] = performance_6['estimator'].map(estimator_names)

# 1.1 Box Plot of Standard Deviation for 6 Angles
box_data_6 = []
for estimator in ['mean', 'median', 'trimmed_mean_20', 'midmean', 'huber', 'ransac']:
    for value in stability_6[f'{estimator}_std'].dropna():
        box_data_6.append({
            'Estimator': estimator_names[estimator],
            'Std_Deviation': value
        })

box_df_6 = pd.DataFrame(box_data_6)

fig_6_std = px.box(
    box_df_6,
    x='Estimator',
    y='Std_Deviation',
    title="<b>6-Angle Analysis: Standard Deviation of Radius Estimates</b><br><sub>Lower values indicate better stability</sub>",
    labels={'Std_Deviation': 'Standard Deviation (pixels)'},
    color='Estimator',
    color_discrete_sequence=px.colors.qualitative.Set2
)

fig_6_std.update_layout(
    xaxis_title="Estimation Method",
    yaxis_title="Standard Deviation (pixels)",
    showlegend=False,
    height=600
)

# 1.2 Parallel Coordinates Plot for 6 Angles
parallel_data_6 = []
for _, row in stability_6.iterrows():
    for estimator in ['mean', 'median', 'trimmed_mean_20', 'midmean', 'huber', 'ransac']:
        parallel_data_6.append({
            'Person_Eye': f"{row['person_id']}_{row['eye']}",
            'Estimator': estimator_names[estimator],
            'Std_Deviation': row[f'{estimator}_std'],
            'MAD': row[f'{estimator}_mad'],
            'IQR': row[f'{estimator}_iqr'],
            'Mean_Radius': row[f'{estimator}_mean']
        })

parallel_df_6 = pd.DataFrame(parallel_data_6)

fig_6_parallel = px.parallel_coordinates(
    parallel_df_6,
    dimensions=['Std_Deviation', 'MAD', 'IQR', 'Mean_Radius'],
    color='Std_Deviation',
    color_continuous_scale=px.colors.sequential.Viridis,
    title="<b>6-Angle Analysis: Parallel Coordinates of All Metrics</b><br><sub>Each line represents one person's eye across different estimators</sub>"
)

# 1.3 3D Scatter Plot for 6 Angles
fig_6_3d = px.scatter_3d(
    performance_6,
    x='avg_std',
    y='avg_mad', 
    z='avg_iqr',
    color='estimator_label',
    size='num_eyes',
    hover_name='estimator_label',
    title="<b>6-Angle Analysis: 3D Performance Space</b><br><sub>Size represents number of eyes analyzed</sub>",
    labels={'avg_std': 'Avg Std Dev', 'avg_mad': 'Avg MAD', 'avg_iqr': 'Avg IQR'}
)

fig_6_3d.update_traces(marker=dict(symbol='circle', sizemode='diameter'))

# 1.4 Radar Chart for 6 Angles
fig_6_radar = create_radar_chart(performance_6, "<b>6-Angle Analysis: Performance Radar Chart</b><br><sub>Higher values are better in all dimensions</sub>")

# ========== PHASE 2: 8-ANGLES INTERACTIVE VISUALIZATION ==========

print("Creating 8-angle interactive visualizations...")
stability_8, performance_8, df_8 = analyze_angle_set_interactive(df, angle_columns_8, "8 Angles")
performance_8['estimator_label'] = performance_8['estimator'].map(estimator_names)

# 2.1 Box Plot of Standard Deviation for 8 Angles
box_data_8 = []
for estimator in ['mean', 'median', 'trimmed_mean_20', 'midmean', 'huber', 'ransac']:
    for value in stability_8[f'{estimator}_std'].dropna():
        box_data_8.append({
            'Estimator': estimator_names[estimator],
            'Std_Deviation': value
        })

box_df_8 = pd.DataFrame(box_data_8)

fig_8_std = px.box(
    box_df_8,
    x='Estimator',
    y='Std_Deviation',
    title="<b>8-Angle Analysis: Standard Deviation of Radius Estimates</b><br><sub>Lower values indicate better stability</sub>",
    labels={'Std_Deviation': 'Standard Deviation (pixels)'},
    color='Estimator',
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig_8_std.update_layout(
    xaxis_title="Estimation Method",
    yaxis_title="Standard Deviation (pixels)",
    showlegend=False,
    height=600
)

# 2.2 Violin Plot for 8 Angles
fig_8_violin = px.violin(
    box_df_8,
    x='Estimator',
    y='Std_Deviation',
    color='Estimator',
    title="<b>8-Angle Analysis: Distribution of Standard Deviations</b><br><sub>Violin plots show probability density</sub>",
    box=True,
    points="all"
)

fig_8_violin.update_layout(
    height=600,
    showlegend=False
)

# 2.3 Heatmap of Correlation Between Metrics for 8 Angles
correlation_metrics_8 = stability_8[['mean_std', 'median_std', 'trimmed_mean_20_std', 'midmean_std', 'huber_std', 'ransac_std']]
correlation_metrics_8.columns = [estimator_names[col.replace('_std', '')] for col in correlation_metrics_8.columns]
corr_matrix_8 = correlation_metrics_8.corr()

fig_8_heatmap = px.imshow(
    corr_matrix_8,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title="<b>8-Angle Analysis: Correlation Between Methods</b><br><sub>How different estimators relate to each other</sub>"
)

fig_8_heatmap.update_layout(height=600)

# 2.4 Radar Chart for 8 Angles
fig_8_radar = create_radar_chart(performance_8, "<b>8-Angle Analysis: Performance Radar Chart</b><br><sub>Higher values are better in all dimensions</sub>")

# ========== PHASE 3: COMPREHENSIVE COMPARISON VISUALIZATION ==========

print("Creating comprehensive comparison visualizations...")

# 3.1 Side-by-Side Comparison Bar Chart
comparison_data = []
for estimator in ['mean', 'median', 'trimmed_mean_20', 'midmean', 'huber', 'ransac']:
    # Find matching performance data
    perf_6_match = performance_6[performance_6['estimator'] == estimator]
    perf_8_match = performance_8[performance_8['estimator'] == estimator]
    
    if len(perf_6_match) > 0 and len(perf_8_match) > 0:
        perf_6 = perf_6_match.iloc[0]
        perf_8 = perf_8_match.iloc[0]
        
        comparison_data.extend([
            {'Estimator': estimator_names[estimator], 'Angle_Set': '6 Angles', 'Std_Deviation': perf_6['avg_std']},
            {'Estimator': estimator_names[estimator], 'Angle_Set': '8 Angles', 'Std_Deviation': perf_8['avg_std']}
        ])

comparison_df = pd.DataFrame(comparison_data)

fig_comparison_bar = px.bar(
    comparison_df,
    x='Estimator',
    y='Std_Deviation',
    color='Angle_Set',
    barmode='group',
    title="<b>Direct Comparison: 6 vs 8 Angles</b><br><sub>Average Standard Deviation by Method</sub>",
    color_discrete_sequence=['#1f77b4', '#ff7f0e'],
    text_auto='.2f'
)

fig_comparison_bar.update_layout(
    yaxis_title="Average Standard Deviation (pixels)",
    height=600
)

# 3.2 Improvement Percentage Gauge Chart
improvement_data = []
for estimator in ['mean', 'median', 'trimmed_mean_20', 'midmean', 'huber', 'ransac']:
    perf_6_match = performance_6[performance_6['estimator'] == estimator]
    perf_8_match = performance_8[performance_8['estimator'] == estimator]
    
    if len(perf_6_match) > 0 and len(perf_8_match) > 0:
        perf_6 = perf_6_match.iloc[0]
        perf_8 = perf_8_match.iloc[0]
        
        # Safe division with error handling
        if perf_6['avg_std'] > 0:
            improvement_pct = ((perf_6['avg_std'] - perf_8['avg_std']) / perf_6['avg_std']) * 100
        else:
            improvement_pct = 0
        
        improvement_data.append({
            'Estimator': estimator_names[estimator],
            'Improvement_Percentage': improvement_pct
        })

improvement_df = pd.DataFrame(improvement_data)

# Create gauge chart subplots
if len(improvement_df) > 0:
    fig_gauges = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
        subplot_titles=[f"<b>{est}</b>" for est in improvement_df['Estimator']]
    )

    for i, (_, row) in enumerate(improvement_df.iterrows()):
        row_idx = i // 3 + 1
        col_idx = i % 3 + 1
        
        # Determine color based on improvement
        if row['Improvement_Percentage'] > 5:
            gauge_color = 'green'
        elif row['Improvement_Percentage'] > 0:
            gauge_color = 'yellow'
        else:
            gauge_color = 'red'
        
        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=row['Improvement_Percentage'],
                domain={'row': row_idx-1, 'column': col_idx-1},
                title={'text': f"{row['Estimator']}"},
                delta={'reference': 0, 'suffix': '%'},
                gauge={
                    'axis': {'range': [-20, 20]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [-20, 0], 'color': "lightgray"},
                        {'range': [0, 5], 'color': "lightyellow"},
                        {'range': [5, 20], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ),
            row=row_idx, col=col_idx
        )

    fig_gauges.update_layout(
        height=500,
        title_text="<b>Improvement Percentage with 8 Angles</b><br><sub>Positive values indicate improvement</sub>"
    )
else:
    fig_gauges = go.Figure()
    fig_gauges.add_annotation(text="No improvement data available", x=0.5, y=0.5, showarrow=False)

# 3.3 Fixed Interactive Scatter Matrix for Comparison
scatter_comparison_data = []
for estimator in ['mean', 'median', 'trimmed_mean_20', 'midmean', 'huber', 'ransac']:
    for _, row_6 in stability_6.iterrows():
        person_eye = f"{row_6['person_id']}_{row_6['eye']}"
        std_6 = row_6[f'{estimator}_std']
        
        # Find matching 8-angle data
        matching_8 = stability_8[
            (stability_8['person_id'] == row_6['person_id']) & 
            (stability_8['eye'] == row_6['eye'])
        ]
        
        if not matching_8.empty:
            std_8 = matching_8.iloc[0][f'{estimator}_std']
            improvement = std_6 - std_8
            
            # Use absolute value for size to avoid negative values
            size_value = abs(improvement) + 1  # +1 to ensure positive size
            
            scatter_comparison_data.append({
                'Estimator': estimator_names[estimator],
                'Person_Eye': person_eye,
                'Std_6_Angles': std_6,
                'Std_8_Angles': std_8,
                'Improvement': improvement,
                'Size': size_value,
                'Color': 'green' if improvement > 0 else 'red'
            })

scatter_comparison_df = pd.DataFrame(scatter_comparison_data)

if len(scatter_comparison_df) > 0:
    fig_scatter_matrix = px.scatter(
        scatter_comparison_df,
        x='Std_6_Angles',
        y='Std_8_Angles',
        color='Improvement',  # Use improvement for color instead of estimator
        size='Size',  # Use the calculated size
        hover_data=['Person_Eye', 'Estimator'],
        title="<b>6 vs 8 Angles: Individual Eye Comparison</b><br><sub>Points below the diagonal line show improvement with 8 angles</sub>",
        labels={'Std_6_Angles': '6-Angle Std Dev', 'Std_8_Angles': '8-Angle Std Dev'},
        color_continuous_scale='RdYlGn',  # Red-Yellow-Green scale
        color_continuous_midpoint=0  # Center at 0 improvement
    )

    # Add diagonal line
    max_val = scatter_comparison_df[['Std_6_Angles', 'Std_8_Angles']].max().max()
    fig_scatter_matrix.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='No Change Line'
        )
    )

    fig_scatter_matrix.update_layout(height=700)
else:
    fig_scatter_matrix = go.Figure()
    fig_scatter_matrix.add_annotation(text="No comparison data available", x=0.5, y=0.5, showarrow=False)

# ========== SAVE ALL VISUALIZATIONS ==========

print("Saving all visualizations as HTML files...")

# Save 6-angle visualizations
fig_6_std.write_html("visualizations/6_angles_std_deviation.html")
fig_6_parallel.write_html("visualizations/6_angles_parallel_coordinates.html")
fig_6_3d.write_html("visualizations/6_angles_3d_scatter.html")
fig_6_radar.write_html("visualizations/6_angles_radar_chart.html")

# Save 8-angle visualizations
fig_8_std.write_html("visualizations/8_angles_std_deviation.html")
fig_8_violin.write_html("visualizations/8_angles_violin_plot.html")
fig_8_heatmap.write_html("visualizations/8_angles_correlation_heatmap.html")
fig_8_radar.write_html("visualizations/8_angles_radar_chart.html")

# Save comparison visualizations
fig_comparison_bar.write_html("visualizations/comparison_bar_chart.html")
fig_gauges.write_html("visualizations/improvement_gauges.html")
fig_scatter_matrix.write_html("visualizations/scatter_matrix_comparison.html")

print("✅ All interactive visualizations created and saved!")
print("📊 Open the HTML files in the 'visualizations' folder in your browser")
print("🎯 You can zoom, pan, hover for details, and interact with all charts")

# ========== FINAL SUMMARY ==========

# Calculate overall improvement with safe division
if len(performance_6) > 0 and len(performance_8) > 0:
    avg_std_6 = performance_6['avg_std'].mean()
    avg_std_8 = performance_8['avg_std'].mean()
    
    if avg_std_6 > 0:
        overall_improvement = ((avg_std_6 - avg_std_8) / avg_std_6) * 100
    else:
        overall_improvement = 0

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Average Standard Deviation:")
    print(f"  6 Angles: {avg_std_6:.3f} pixels")
    print(f"  8 Angles: {avg_std_8:.3f} pixels")
    print(f"  Overall Improvement: {overall_improvement:+.2f}%")
    print(f"{'='*60}")

    # Find best method for each set
    if len(performance_6) > 0:
        best_6 = performance_6.loc[performance_6['avg_std'].idxmin()]
        print(f"Best 6-Angle Method: {estimator_names[best_6['estimator']]} ({best_6['avg_std']:.3f} px)")
    
    if len(performance_8) > 0:
        best_8 = performance_8.loc[performance_8['avg_std'].idxmin()]
        print(f"Best 8-Angle Method: {estimator_names[best_8['estimator']]} ({best_8['avg_std']:.3f} px)")

    if len(performance_6) > 0 and len(performance_8) > 0:
        improvement_best = ((best_6['avg_std'] - best_8['avg_std']) / best_6['avg_std']) * 100
        print(f"Improvement in Best Method: {improvement_best:+.2f}%")
else:
    print("No performance data available for summary")

# Save performance data
if len(performance_6) > 0:
    performance_6.to_excel("visualizations/performance_6_angles.xlsx", index=False)
if len(performance_8) > 0:
    performance_8.to_excel("visualizations/performance_8_angles.xlsx", index=False)

print(f"\n📁 All files saved in the 'visualizations' folder:")
print("   - 12 interactive HTML charts")
print("   - 2 Excel files with performance data")
print("   - Open any .html file in your web browser to explore the visualizations")
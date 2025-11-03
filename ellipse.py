# -*- coding: utf-8 -*-
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import HuberRegressor  # robust location via intercept_

# ---------------------------------------------------------------------
# Paths and data
# ---------------------------------------------------------------------
os.makedirs('visulization_ellipse', exist_ok=True)
df = pd.read_excel(r"D:\SEM-5\ELECTIVE'S\DEPARTMENT-ELECTIVE\BIOMETRICS\Project\iris_measurements.xlsx")

print("Creating interactive visulization_ellipse...")

# Angle sets
angles_6 = [30, 0, -30, 150, 180, 210]
angles_8 = [30, 0, -30, 150, 180, 210, 90, 270]

angle_columns_6 = [f'radius_{a}' for a in angles_6]
angle_columns_8 = [f'radius_{a}' for a in angles_8]

# Estimators to use (RANSAC removed)
ESTIMATORS = ['mean', 'median', 'trimmed_mean_20', 'midmean', 'huber']
estimator_names = {
    'mean': 'Mean',
    'median': 'Median',
    'trimmed_mean_20': 'Trimmed Mean (20%)',
    'midmean': 'Midmean',
    'huber': 'Huber'
}

# ---------------------------------------------------------------------
# Ellipse fitting utilities (Fitzgibbon et al. 1999)
#   Fits conic: A x^2 + B x y + C y^2 + D x + E y + F = 0
#   Constraint: 4 A C - B^2 = 1 ensures an ellipse
# ---------------------------------------------------------------------
def fit_ellipse_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 5:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    D = np.vstack([x*x, x*y, y*y, x, y, np.ones_like(x)]).T  # (N,6)
    S = D.T @ D                                               # (6,6)

    # Constraint matrix for ellipse: a^T C a = 1 with 4AC - B^2 = 1
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1

    try:
        # Solve generalized eigenproblem: inv(S) C a = λ a
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S) @ C)
        # Select eigenvector that yields an ellipse: 4AC - B^2 > 0
        best = None
        best_val = -np.inf
        for i in range(eigvals.size):
            a = np.real(eigvecs[:, i])
            A, B, Cc, Dd, Ee, Ff = a
            cond = 4*A*Cc - B*B
            if cond > best_val:
                best_val = cond
                best = a
        if best is None:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        A, B, Cc, Dd, Ee, Ff = best
        denom = B*B - 4*A*Cc
        if abs(denom) < 1e-12:
            return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        # Center
        x0 = (2*Cc*Dd - B*Ee) / denom
        y0 = (2*A*Ee - B*Dd) / denom

        # Translate to center to derive axes lengths
        up = 2*(A*Ee*Ee + Cc*Dd*Dd + Ff*B*B/2 - B*Dd*Ee - A*Cc*Ff)
        term = np.sqrt((A - Cc)**2 + B*B)
        down1 = denom * ((A + Cc) + term)
        down2 = denom * ((A + Cc) - term)
        if down1 == 0 or down2 == 0:
            return (np.nan, np.nan, np.nan, x0, y0, np.nan)

        a_len = np.sqrt(abs(up / down1))
        b_len = np.sqrt(abs(up / down2))

        # Orientation
        phi = 0.5 * np.arctan2(B, (A - Cc))

        # Ensure a_len >= b_len
        if b_len > a_len:
            a_len, b_len = b_len, a_len
            phi = (phi + np.pi/2) % np.pi

        ecc = np.sqrt(max(0.0, 1.0 - (b_len*b_len) / (a_len*a_len)))
        return (a_len, b_len, phi, x0, y0, ecc)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

def ellipse_from_polar(radii, angles_deg):
    theta = np.deg2rad(np.asarray(angles_deg, dtype=float))
    x = radii * np.cos(theta)
    y = radii * np.sin(theta)
    return fit_ellipse_xy(x, y)

# ---------------------------------------------------------------------
# Analysis function (RANSAC removed; Huber fixed to use intercept_)
# ---------------------------------------------------------------------
def analyze_angle_set_interactive(df, angle_columns, set_name):
    """Compute estimators per image, ellipse parameters, and per-eye stability metrics."""

    # For labeling ellipse inputs later
    angle_values = [int(c.split('_')[1]) for c in angle_columns]

    def apply_estimators(row, angles):
        radii = row[angles].values.astype(float)
        res = {}
        # Basic robust/central estimates
        res['mean'] = float(np.mean(radii))
        res['median'] = float(np.median(radii))
        res['trimmed_mean_20'] = float(stats.trim_mean(radii, proportiontocut=0.2))
        res['midmean'] = float(np.mean(np.percentile(radii, [25, 75])))

        # Huber robust location: use intercept_ (with X all zeros)
        X0 = np.zeros((len(radii), 1))
        try:
            huber = HuberRegressor(epsilon=1.35, alpha=0.0, fit_intercept=True, max_iter=200)
            huber.fit(X0, radii)
            res['huber'] = float(huber.intercept_)  # robust center
            res['huber_success'] = 1
        except Exception:
            res['huber'] = float(np.median(radii))
            res['huber_success'] = 0

        # Ellipse parameters from current sample set
        try:
            a_len, b_len, phi, x0, y0, ecc = ellipse_from_polar(radii, angle_values)
            res.update({
                'ellipse_a': float(a_len),
                'ellipse_b': float(b_len),
                'ellipse_phi_deg': float(np.rad2deg(phi)) if np.isfinite(phi) else np.nan,
                'ellipse_ecc': float(ecc),
                'ellipse_area': float(np.pi * a_len * b_len) if np.isfinite(a_len) and np.isfinite(b_len) else np.nan
            })
        except Exception:
            res.update({'ellipse_a': np.nan, 'ellipse_b': np.nan, 'ellipse_phi_deg': np.nan,
                        'ellipse_ecc': np.nan, 'ellipse_area': np.nan})
        return pd.Series(res)

    # Per-image estimators and ellipse
    est_res = df.apply(lambda r: apply_estimators(r, angle_columns), axis=1)
    df_temp = pd.concat([df.copy(), est_res], axis=1)

    # Per-eye stability metrics (only for eyes with >1 images)
    stability_rows = []
    for (pid, eye), grp in df_temp.groupby(['person_id', 'eye']):
        if len(grp) <= 1:
            continue
        eye_s = {'person_id': pid, 'eye': eye, 'num_images': len(grp)}
        for est in ESTIMATORS:
            arr = grp[est].values.astype(float)
            mad = np.median(np.abs(arr - np.median(arr)))
            std = float(np.std(arr))
            if std == 0:
                std = 0.001
            eye_s.update({
                f'{est}_std': std,
                f'{est}_mad': float(mad),
                f'{est}_iqr': float(stats.iqr(arr)),
                f'{est}_mean': float(np.mean(arr)),
                f'{est}_min': float(np.min(arr)),
                f'{est}_max': float(np.max(arr)),
                f'{est}_range': float(np.ptp(arr))
            })
        # Aggregate ellipse stats (means)
        for k in ['ellipse_a', 'ellipse_b', 'ellipse_ecc', 'ellipse_area', 'ellipse_phi_deg']:
            if k in grp:
                eye_s[f'{k}_mean'] = float(np.nanmean(grp[k].values)) if grp[k].notna().any() else np.nan
                eye_s[f'{k}_std'] = float(np.nanstd(grp[k].values)) if grp[k].notna().any() else np.nan
        stability_rows.append(eye_s)

    stability_df = pd.DataFrame(stability_rows)

    # Estimator-level performance summary
    perf_rows = []
    for est in ESTIMATORS:
        if f'{est}_std' in stability_df:
            std_vals = stability_df[f'{est}_std'].dropna()
            mad_vals = stability_df.get(f'{est}_mad', pd.Series(dtype=float)).dropna()
            iqr_vals = stability_df.get(f'{est}_iqr', pd.Series(dtype=float)).dropna()
            if len(std_vals) > 0:
                perf_rows.append({
                    'estimator': est,
                    'set_name': set_name,
                    'avg_std': float(std_vals.mean()),
                    'avg_mad': float(mad_vals.mean()) if len(mad_vals) else np.nan,
                    'avg_iqr': float(iqr_vals.mean()) if len(iqr_vals) else np.nan,
                    'min_std': float(std_vals.min()),
                    'max_std': float(std_vals.max()),
                    'std_of_std': float(std_vals.std()),
                    'num_eyes': int(len(std_vals))
                })
    performance_df = pd.DataFrame(perf_rows)
    return stability_df, performance_df, df_temp

# ---------------------------------------------------------------------
# Radar chart helper
# ---------------------------------------------------------------------
def create_radar_chart(performance_df, title):
    categories = ['Stability (1/Std)', 'Robustness (1/MAD)', 'Consistency (1/IQR)']
    fig = go.Figure()
    for _, row in performance_df.iterrows():
        stability = 1 / row['avg_std'] if row['avg_std'] > 0 else 0
        robustness = 1 / row['avg_mad'] if (pd.notna(row['avg_mad']) and row['avg_mad'] > 0) else 0
        consistency = 1 / row['avg_iqr'] if (pd.notna(row['avg_iqr']) and row['avg_iqr'] > 0) else 0
        max_val = max(stability, robustness, consistency, 1e-9)
        r = [stability/max_val, robustness/max_val, consistency/max_val, stability/max_val]
        fig.add_trace(go.Scatterpolar(r=r, theta=categories + [categories[0]], fill='toself',
                                      name=row['estimator_label']))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=True, title=title, height=600)
    return fig

# ---------------------------------------------------------------------
# Phase 1: 6-angle analysis
# ---------------------------------------------------------------------
print("Creating 6-angle interactive visulization_ellipse...")
stability_6, performance_6, df_6 = analyze_angle_set_interactive(df, angle_columns_6, "6 Angles")
performance_6['estimator_label'] = performance_6['estimator'].map(estimator_names)

# 6-angle: box of std
box_df_6 = (stability_6
            .loc[:, [f'{e}_std' for e in ESTIMATORS]]
            .melt(var_name='Estimator', value_name='Std_Deviation'))
box_df_6['Estimator'] = box_df_6['Estimator'].str.replace('_std', '', regex=False).map(estimator_names)

fig_6_std = px.box(box_df_6, x='Estimator', y='Std_Deviation',
                   title="<b>6-Angle Analysis: Standard Deviation of Radius Estimates</b><br><sub>Lower values indicate better stability</sub>",
                   labels={'Std_Deviation': 'Standard Deviation (pixels)'},
                   color='Estimator', color_discrete_sequence=px.colors.qualitative.Set2)
fig_6_std.update_layout(xaxis_title="Estimation Method", yaxis_title="Standard Deviation (pixels)", showlegend=False, height=600)

# 6-angle: parallel coordinates of metrics
parallel_df_6 = []
for _, r in stability_6.iterrows():
    for e in ESTIMATORS:
        parallel_df_6.append({
            'Person_Eye': f"{r['person_id']}_{r['eye']}",
            'Estimator': estimator_names[e],
            'Std_Deviation': r[f'{e}_std'],
            'MAD': r[f'{e}_mad'],
            'IQR': r[f'{e}_iqr'],
            'Mean_Radius': r[f'{e}_mean']
        })
parallel_df_6 = pd.DataFrame(parallel_df_6)

fig_6_parallel = px.parallel_coordinates(
    parallel_df_6,
    dimensions=['Std_Deviation', 'MAD', 'IQR', 'Mean_Radius'],
    color='Std_Deviation',
    color_continuous_scale=px.colors.sequential.Viridis,
    title="<b>6-Angle Analysis: Parallel Coordinates of All Metrics</b><br><sub>Each line represents one person's eye across different estimators</sub>"
)

# 6-angle: 3D estimator performance
fig_6_3d = px.scatter_3d(
    performance_6, x='avg_std', y='avg_mad', z='avg_iqr',
    color='estimator_label', size='num_eyes', hover_name='estimator_label',
    title="<b>6-Angle Analysis: 3D Performance Space</b><br><sub>Size represents number of eyes analyzed</sub>",
    labels={'avg_std': 'Avg Std Dev', 'avg_mad': 'Avg MAD', 'avg_iqr': 'Avg IQR'}
)
fig_6_3d.update_traces(marker=dict(symbol='circle', sizemode='diameter'))

# 6-angle: radar
fig_6_radar = create_radar_chart(performance_6, "<b>6-Angle Analysis: Performance Radar Chart</b><br><sub>Higher values are better in all dimensions</sub>")

# 6-angle: ellipse diagnostics (per image)
ellipse_cols = ['person_id', 'eye', 'ellipse_a', 'ellipse_b', 'ellipse_ecc', 'ellipse_area', 'ellipse_phi_deg']
ellipse_6 = df_6[ellipse_cols].dropna()
fig_ellipse_6 = px.scatter(
    ellipse_6, x='ellipse_a', y='ellipse_b', color='ellipse_ecc',
    hover_data=['person_id', 'eye', 'ellipse_area', 'ellipse_phi_deg'],
    title="6-Angle: Ellipse Axes Scatter (a vs b)",
    labels={'ellipse_a': 'Major axis (px)', 'ellipse_b': 'Minor axis (px)', 'ellipse_ecc': 'Eccentricity'}
)

# ---------------------------------------------------------------------
# Phase 2: 8-angle analysis
# ---------------------------------------------------------------------
print("Creating 8-angle interactive visulization_ellipse...")
stability_8, performance_8, df_8 = analyze_angle_set_interactive(df, angle_columns_8, "8 Angles")
performance_8['estimator_label'] = performance_8['estimator'].map(estimator_names)

# 8-angle: box of std
box_df_8 = (stability_8
            .loc[:, [f'{e}_std' for e in ESTIMATORS]]
            .melt(var_name='Estimator', value_name='Std_Deviation'))
box_df_8['Estimator'] = box_df_8['Estimator'].str.replace('_std', '', regex=False).map(estimator_names)
fig_8_std = px.box(
    box_df_8, x='Estimator', y='Std_Deviation',
    title="<b>8-Angle Analysis: Standard Deviation of Radius Estimates</b><br><sub>Lower values indicate better stability</sub>",
    labels={'Std_Deviation': 'Standard Deviation (pixels)'},
    color='Estimator', color_discrete_sequence=px.colors.qualitative.Set3
)
fig_8_std.update_layout(xaxis_title="Estimation Method", yaxis_title="Standard Deviation (pixels)", showlegend=False, height=600)

# 8-angle: violin of std distributions
fig_8_violin = px.violin(
    box_df_8, x='Estimator', y='Std_Deviation', color='Estimator',
    title="<b>8-Angle Analysis: Distribution of Standard Deviations</b><br><sub>Violin plots show probability density</sub>",
    box=True, points="all"
)
fig_8_violin.update_layout(height=600, showlegend=False)

# 8-angle: correlation heatmap across methods (no RANSAC)
corr_cols = ['mean_std', 'median_std', 'trimmed_mean_20_std', 'midmean_std', 'huber_std']
corr_named = [estimator_names[c.replace('_std', '')] for c in corr_cols]
corr_df = stability_8[corr_cols].copy()
corr_df.columns = corr_named
corr_matrix_8 = corr_df.corr()
fig_8_heatmap = px.imshow(
    corr_matrix_8, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r',
    title="<b>8-Angle Analysis: Correlation Between Methods</b><br><sub>How different estimators relate to each other</sub>"
)
fig_8_heatmap.update_layout(height=600)

# 8-angle: radar
fig_8_radar = create_radar_chart(performance_8, "<b>8-Angle Analysis: Performance Radar Chart</b><br><sub>Higher values are better in all dimensions</sub>")

# 8-angle: ellipse diagnostics
ellipse_8 = df_8[ellipse_cols].dropna()
fig_ellipse_8 = px.scatter(
    ellipse_8, x='ellipse_a', y='ellipse_b', color='ellipse_ecc',
    hover_data=['person_id', 'eye', 'ellipse_area', 'ellipse_phi_deg'],
    title="8-Angle: Ellipse Axes Scatter (a vs b)",
    labels={'ellipse_a': 'Major axis (px)', 'ellipse_b': 'Minor axis (px)', 'ellipse_ecc': 'Eccentricity'}
)

# Optional: Eccentricity histograms for quick quality check
fig_ecc_hist = px.histogram(
    pd.concat([ellipse_6.assign(Angle_Set='6 Angles'), ellipse_8.assign(Angle_Set='8 Angles')], ignore_index=True),
    x='ellipse_ecc', color='Angle_Set', barmode='overlay', nbins=30,
    title="Eccentricity Distribution: 6 vs 8 Angles",
    labels={'ellipse_ecc': 'Eccentricity'}
)

# ---------------------------------------------------------------------
# Phase 3: Comparison visulization_ellipse (no RANSAC)
# ---------------------------------------------------------------------
print("Creating comprehensive comparison visulization_ellipse...")

# 3.1 Side-by-side bar: 6 vs 8 average std per method
comparison_rows = []
for e in ESTIMATORS:
    p6 = performance_6.loc[performance_6['estimator'] == e]
    p8 = performance_8.loc[performance_8['estimator'] == e]
    if len(p6) and len(p8):
        comparison_rows.extend([
            {'Estimator': estimator_names[e], 'Angle_Set': '6 Angles', 'Std_Deviation': float(p6.iloc[0]['avg_std'])},
            {'Estimator': estimator_names[e], 'Angle_Set': '8 Angles', 'Std_Deviation': float(p8.iloc[0]['avg_std'])}
        ])
comparison_df = pd.DataFrame(comparison_rows)
fig_comparison_bar = px.bar(
    comparison_df, x='Estimator', y='Std_Deviation', color='Angle_Set', barmode='group',
    title="<b>Direct Comparison: 6 vs 8 Angles</b><br><sub>Average Standard Deviation by Method</sub>",
    color_discrete_sequence=['#1f77b4', '#ff7f0e'], text_auto='.2f'
)
fig_comparison_bar.update_layout(yaxis_title="Average Standard Deviation (pixels)", height=600)

# 3.2 Improvement percentage gauges (8 vs 6)
imp_rows = []
for e in ESTIMATORS:
    p6 = performance_6.loc[performance_6['estimator'] == e]
    p8 = performance_8.loc[performance_8['estimator'] == e]
    if len(p6) and len(p8) and p6.iloc[0]['avg_std'] > 0:
        improve_pct = ((p6.iloc[0]['avg_std'] - p8.iloc[0]['avg_std']) / p6.iloc[0]['avg_std']) * 100.0
        imp_rows.append({'Estimator': estimator_names[e], 'Improvement_Percentage': float(improve_pct)})
improvement_df = pd.DataFrame(imp_rows)

if len(improvement_df) > 0:
    fig_gauges = make_subplots(
        rows=2, cols=3,
        specs=[[{'type': 'indicator'}]*3, [{'type': 'indicator'}]*3],
        subplot_titles=[f"<b>{est}</b>" for est in improvement_df['Estimator']]
    )
    for i, (_, row) in enumerate(improvement_df.iterrows()):
        r = i // 3 + 1
        c = i % 3 + 1
        val = row['Improvement_Percentage']
        gauge_color = 'green' if val > 5 else ('yellow' if val > 0 else 'red')
        fig_gauges.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=val,
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
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0}
                }
            ),
            row=r, col=c
        )
    fig_gauges.update_layout(height=500, title_text="<b>Improvement Percentage with 8 Angles</b><br><sub>Positive values indicate improvement</sub>")
else:
    fig_gauges = go.Figure()
    fig_gauges.add_annotation(text="No improvement data available", x=0.5, y=0.5, showarrow=False)

# 3.3 Eye-level scatter comparison (6 vs 8) for each method
scatter_rows = []
for e in ESTIMATORS:
    for _, r6 in stability_6.iterrows():
        match = stability_8[(stability_8['person_id'] == r6['person_id']) & (stability_8['eye'] == r6['eye'])]
        if not match.empty:
            std_6 = float(r6[f'{e}_std'])
            std_8 = float(match.iloc[0][f'{e}_std'])
            diff = std_6 - std_8
            scatter_rows.append({
                'Estimator': estimator_names[e],
                'Person_Eye': f"{r6['person_id']}_{r6['eye']}",
                'Std_6_Angles': std_6,
                'Std_8_Angles': std_8,
                'Improvement': diff,
                'Size': abs(diff) + 1.0
            })
scatter_df = pd.DataFrame(scatter_rows)
if len(scatter_df) > 0:
    fig_scatter_matrix = px.scatter(
        scatter_df, x='Std_6_Angles', y='Std_8_Angles',
        color='Improvement', size='Size', hover_data=['Person_Eye', 'Estimator'],
        title="<b>6 vs 8 Angles: Individual Eye Comparison</b><br><sub>Points below the diagonal line show improvement with 8 angles</sub>",
        labels={'Std_6_Angles': '6-Angle Std Dev', 'Std_8_Angles': '8-Angle Std Dev'},
        color_continuous_scale='RdYlGn', color_continuous_midpoint=0
    )
    max_val = scatter_df[['Std_6_Angles', 'Std_8_Angles']].to_numpy().max()
    fig_scatter_matrix.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                                            line=dict(color='red', dash='dash'), name='No Change Line'))
    fig_scatter_matrix.update_layout(height=700)
else:
    fig_scatter_matrix = go.Figure()
    fig_scatter_matrix.add_annotation(text="No comparison data available", x=0.5, y=0.5, showarrow=False)

# ---------------------------------------------------------------------
# Save all visulization_ellipse
# ---------------------------------------------------------------------
print("Saving all visulization_ellipse as HTML files...")

# 6-angle
fig_6_std.write_html("visulization_ellipse/6_angles_std_deviation.html")
fig_6_parallel.write_html("visulization_ellipse/6_angles_parallel_coordinates.html")
fig_6_3d.write_html("visulization_ellipse/6_angles_3d_scatter.html")
fig_6_radar.write_html("visulization_ellipse/6_angles_radar_chart.html")
fig_ellipse_6.write_html("visulization_ellipse/6_angles_ellipse_axes_scatter.html")

# 8-angle
fig_8_std.write_html("visulization_ellipse/8_angles_std_deviation.html")
fig_8_violin.write_html("visulization_ellipse/8_angles_violin_plot.html")
fig_8_heatmap.write_html("visulization_ellipse/8_angles_correlation_heatmap.html")
fig_8_radar.write_html("visulization_ellipse/8_angles_radar_chart.html")
fig_ellipse_8.write_html("visulization_ellipse/8_angles_ellipse_axes_scatter.html")
fig_ecc_hist.write_html("visulization_ellipse/ellipse_eccentricity_hist.html")

# Comparisons
fig_comparison_bar.write_html("visulization_ellipse/comparison_bar_chart.html")
fig_gauges.write_html("visulization_ellipse/improvement_gauges.html")
fig_scatter_matrix.write_html("visulization_ellipse/scatter_matrix_comparison.html")

print("✅ All interactive visulization_ellipse created and saved!")
print("📊 Open the HTML files in the 'visulization_ellipse' folder in your browser")
print("🎯 You can zoom, pan, hover for details, and interact with all charts")

# ---------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------
if len(performance_6) and len(performance_8):
    avg_std_6 = float(performance_6['avg_std'].mean())
    avg_std_8 = float(performance_8['avg_std'].mean())
    overall_improvement = ((avg_std_6 - avg_std_8) / avg_std_6) * 100.0 if avg_std_6 > 0 else 0.0

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Overall Average Standard Deviation:\n  6 Angles: {avg_std_6:.3f} px\n  8 Angles: {avg_std_8:.3f} px\n  Overall Improvement: {overall_improvement:+.2f}%")
    if len(performance_6):
        best_6 = performance_6.loc[performance_6['avg_std'].idxmin()]
        print(f"Best 6-Angle Method: {estimator_names[best_6['estimator']]} ({best_6['avg_std']:.3f} px)")
    if len(performance_8):
        best_8 = performance_8.loc[performance_8['avg_std'].idxmin()]
        print(f"Best 8-Angle Method: {estimator_names[best_8['estimator']]} ({best_8['avg_std']:.3f} px)")
    if len(performance_6) and len(performance_8):
        impr_best = ((best_6['avg_std'] - best_8['avg_std']) / best_6['avg_std']) * 100.0
        print(f"Improvement in Best Method: {impr_best:+.2f}%")

# Save performance tables
if len(performance_6):
    performance_6.to_excel("visulization_ellipse/performance_6_angles.xlsx", index=False)
if len(performance_8):
    performance_8.to_excel("visulization_ellipse/performance_8_angles.xlsx", index=False)

print("\n📁 All files saved in the 'visulization_ellipse' folder:")
print("   - 13+ interactive HTML charts (including ellipse diagnostics)")
print("   - 2 Excel files with performance data")
print("   - Open any .html file in your web browser to explore the visulization_ellipse")

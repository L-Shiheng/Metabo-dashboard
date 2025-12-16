import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ==========================================
# 1. 页面基本配置
# ==========================================
st.set_page_config(
    page_title="MetaboAnalyst-Lite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 优化
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    h1 {font-size: 1.8rem !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心计算函数库
# ==========================================

@st.cache_data
def preprocess_data(df, group_col, log_transform=True):
    """
    数据清洗与预处理
    """
    # 提取数值列（代谢物）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 防止分组列被误判为数值列
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)
        
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    # 简单的缺失值填充
    if data_df.isnull().sum().sum() > 0:
        data_df.fillna(data_df.min().min() * 0.5, inplace=True)
    
    if log_transform:
        # Log2(x+1) 避免 log(0)
        data_df = np.log2(data_df + 1)
        
    return pd.concat([meta_df, data_df], axis=1), numeric_cols

# ❌ 已删除 @st.cache_data 以修复 UnhashableParamError
def calculate_vips(model):
    """
    手动计算 PLS-DA 的 VIP 值
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    
    # 矩阵计算
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        
    return vips

@st.cache_data
def run_statistics(df, group_col, case_group, control_group, feature_cols):
    """
    执行单因素统计分析
    """
    group_case = df[df[group_col] == case_group]
    group_ctrl = df[df[group_col] == control_group]
    
    results = []
    
    for feature in feature_cols:
        vals_case = group_case[feature].values
        vals_ctrl = group_ctrl[feature].values
        
        mean_case = np.mean(vals_case)
        mean_ctrl = np.mean(vals_ctrl)
        log2_fc = mean_case - mean_ctrl
        
        try:
            # Welch's t-test
            t_stat, p_val = stats.ttest_ind(vals_case, vals_ctrl, equal_var=False)
        except:
            p_val = 1.0
        
        results.append({
            'Metabolite': feature,
            'Mean_Case': mean_case,
            'Mean_Ctrl': mean_ctrl,
            'Log2_FC': log2_fc,
            'P_Value': p_val
        })
        
    res_df = pd.DataFrame(results)
    
    # FDR 校正
    res_df = res_df.dropna(subset=['P_Value'])
    if not res_df.empty:
        reject, pvals_corrected, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
        res_df['FDR'] = pvals_corrected
        res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
    else:
        res_df['FDR'] = 1.0
        res_df['-Log10_P'] = 0
    
    return res_df

# ==========================================
# 3. 侧边栏：输入与设置
# ==========================================

with st.sidebar:
    st.title("🛠️ 分析设置")
    
    uploaded_file = st.file_uploader("1. 上传 CSV 文件", type=["csv"])
    
    if uploaded_file is None:
        st.info("👋 请上传数据以开始分析。")
        st.markdown("**示例数据格式:**")
        st.markdown("""
        | Sample | Group | Glc | Lac | ... |
        | :--- | :--- | :--- | :--- | :--- |
        | S1 | Cancer | 10.5 | 2.3 | ... |
        | S2 | Healthy| 5.4 | 1.1 | ... |
        """)
        st.stop()
        
    raw_df = pd.read_csv(uploaded_file)
    st.success(f"加载成功: {raw_df.shape[0]} 样本, {raw_df.shape[1]} 列")

    st.divider()

    st.subheader("2. 分组选择")
    non_numeric_cols = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not non_numeric_cols:
        st.error("没有找到分组列（文本列），请检查 CSV。")
        st.stop()
        
    group_col = st.selectbox("选择分组列", non_numeric_cols)
    
    unique_groups = raw_df[group_col].dropna().unique()
    if len(unique_groups) < 2:
        st.error("分组少于 2 个，无法对比。")
        st.stop()
        
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        case_group = st.selectbox("实验组 (Case)", unique_groups, index=0)
    with col_sel2:
        control_group = st.selectbox("对照组 (Ctrl)", unique_groups, index=min(1, len(unique_groups)-1))
        
    if case_group == control_group:
        st.warning("⚠️ 实验组和对照组相同。")
        st.stop()

    st.divider()

    st.subheader("3. 统计参数")
    use_log = st.checkbox("执行 Log2 转换", value=True)
    p_thresh = st.number_input("P-value 阈值", value=0.05, step=0.01, format="%.3f")
    fc_thresh = st.number_input("Log2 FC 阈值", value=1.0, step=0.1)

# ==========================================
# 4. 主逻辑
# ==========================================

# A. 预处理
analysis_df, feature_cols = preprocess_data(raw_df, group_col, log_transform=use_log)
sub_df = analysis_df[analysis_df[group_col].isin([case_group, control_group])].copy()

# B. 统计计算
stats_df = run_statistics(sub_df, group_col, case_group, control_group, feature_cols)

# 标记显著性
def get_sig_label(row):
    if row['P_Value'] < p_thresh and row['Log2_FC'] > fc_thresh:
        return 'Up'
    elif row['P_Value'] < p_thresh and row['Log2_FC'] < -fc_thresh:
        return 'Down'
    else:
        return 'NS'

stats_df['Significant'] = stats_df.apply(get_sig_label, axis=1)
color_map = {'Up': '#E64B35', 'Down': '#3C5488', 'NS': '#B0B0B0'}

# ==========================================
# 5. 结果展示
# ==========================================

st.title("🧪 代谢组学分析报告")
st.markdown(f"**对比**: `{case_group}` vs `{control_group}` | **显著**: `{len(stats_df[stats_df['Significant'] != 'NS'])}` 个")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 PCA 分析", "🎯 PLS-DA 分析", "🌋 火山图", "📦 详情", "📑 数据表"
])

# --- PCA ---
with tab1:
    st.markdown("### PCA 分析")
    col1, col2 = st.columns([3, 1])
    with col1:
        X = sub_df[feature_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
        pca_df['Group'] = sub_df[group_col].values
        
        var = pca.explained_variance_ratio_
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Group',
                         title=f"PCA Score Plot (PC1: {var[0]:.1%}, PC2: {var[1]:.1%})",
                         template="simple_white", width=700, height=500)
        st.plotly_chart(fig, use_container_width=True)

# --- PLS-DA ---
with tab2:
    st.markdown("### PLS-DA 分析")
    col1, col2 = st.columns([3, 1])
    with col1:
        X_pls = sub_df[feature_cols]
        # PLS 也需要 Scaling
        scaler_pls = StandardScaler()
        X_pls_scaled = scaler_pls.fit_transform(X_pls)
        
        y_pls = pd.factorize(sub_df[group_col])[0]
        
        # 建立模型
        pls = PLSRegression(n_components=2)
        pls.fit(X_pls_scaled, y_pls)
        
        # Score Plot
        pls_scores = pd.DataFrame(pls.x_scores_, columns=['Comp 1', 'Comp 2'])
        pls_scores['Group'] = sub_df[group_col].values
        
        fig_pls = px.scatter(pls_scores, x='Comp 1', y='Comp 2', color='Group',
                             title="PLS-DA Score Plot", template="simple_white")
        st.plotly_chart(fig_pls, use_container_width=True)
        
        # VIP Plot
        st.markdown("#### VIP Scores (Top 15)")
        vip_vals = calculate_vips(pls)
        vip_df = pd.DataFrame({'Metabolite': feature_cols, 'VIP': vip_vals})
        vip_df = vip_df.sort_values('VIP', ascending=False).head(15)
        
        fig_vip = px.bar(vip_df, x='VIP', y='Metabolite', orientation='h',
                         color='VIP', template="simple_white", color_continuous_scale='Teal')
        fig_vip.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_vip, use_container_width=True)

# --- Volcano ---
with tab3:
    col1, col2 = st.columns([3, 1])
    with col1:
        fig_vol = px.scatter(stats_df, x="Log2_FC", y="-Log10_P", color="Significant",
                             color_discrete_map=color_map,
                             hover_data=["Metabolite", "P_Value", "FDR"],
                             title="Volcano Plot", template="simple_white")
        fig_vol.add_hline(y=-np.log10(p_thresh), line_dash="dash", line_color="gray")
        fig_vol.add_vline(x=fc_thresh, line_dash="dash", line_color="gray")
        fig_vol.add_vline(x=-fc_thresh, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_vol, use_container_width=True)

# --- Boxplot ---
with tab4:
    sig_feats = stats_df[stats_df['Significant'] != 'NS']['Metabolite'].tolist()
    all_feats = sorted(feature_cols)
    box_feat = st.selectbox("选择代谢物:", sig_feats if sig_feats else all_feats)
    
    if box_feat:
        plot_data = sub_df[[group_col, box_feat]].copy()
        fig_box = px.box(plot_data, x=group_col, y=box_feat, color=group_col,
                         points='all', title=f"{box_feat} 表达量", template="simple_white")
        st.plotly_chart(fig_box, use_container_width=True)

# --- Table ---
with tab5:
    out_df = stats_df.sort_values("P_Value").copy()
    st.dataframe(
        out_df.style.format({
            "Mean_Case": "{:.2f}", "Mean_Ctrl": "{:.2f}",
            "Log2_FC": "{:.2f}", "P_Value": "{:.4f}", 
            "FDR": "{:.4f}", "-Log10_P": "{:.2f}"
        }).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05),
        use_container_width=True
    )
    
    csv_data = out_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 下载 CSV", data=csv_data, file_name="results.csv", mime="text/csv")

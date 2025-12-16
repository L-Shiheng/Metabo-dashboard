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

# 简单的 CSS 优化，减少顶部留白
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
    - 自动识别数值列
    - 可选 Log2 转换
    """
    # 提取数值列（代谢物）和元数据列（分组等）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 排除掉可能被误判为数值的 Group 列（如果是数字编码）
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)
        
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    # 简单的缺失值填充 (用最小值的一半填充，模拟检测限)
    if data_df.isnull().sum().sum() > 0:
        data_df.fillna(data_df.min().min() * 0.5, inplace=True)
    
    if log_transform:
        # Log2(x+1) 避免 log(0)
        data_df = np.log2(data_df + 1)
        
    return pd.concat([meta_df, data_df], axis=1), numeric_cols

@st.cache_data
def calculate_vips(model):
    """
    手动计算 PLS-DA 的 VIP 值 (Variable Importance in Projection)
    Scikit-learn 不直接提供此属性。
    """
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        
    return vips

@st.cache_data
def run_statistics(df, group_col, case_group, control_group, feature_cols):
    """
    执行单因素统计分析: T-test, Fold Change, FDR
    """
    # 提取两组数据
    group_case = df[df[group_col] == case_group]
    group_ctrl = df[df[group_col] == control_group]
    
    results = []
    
    for feature in feature_cols:
        vals_case = group_case[feature].values
        vals_ctrl = group_ctrl[feature].values
        
        # 1. Fold Change (Log2 scale)
        # 假设数据已 Log2 化，差值即为 Log2FC
        mean_case = np.mean(vals_case)
        mean_ctrl = np.mean(vals_ctrl)
        log2_fc = mean_case - mean_ctrl
        
        # 2. T-test (Welch's t-test, 不假设方差相等)
        # 捕获可能的除零错误
        try:
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
    
    # 3. FDR 校正 (Benjamini-Hochberg)
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
    
    # 1. 文件上传
    uploaded_file = st.file_uploader("1. 上传 CSV 文件", type=["csv"], 
                                   help="格式要求：行=样本，列=代谢物，必须包含一列分组名称")
    
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
        
    # 读取数据
    raw_df = pd.read_csv(uploaded_file)
    st.success(f"加载成功: {raw_df.shape[0]} 样本, {raw_df.shape[1]} 列")

    st.divider()

    # 2. 分组设置
    st.subheader("2. 分组选择")
    # 找出所有非数值列作为潜在的分组列
    non_numeric_cols = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not non_numeric_cols:
        st.error("数据中没有找到非数值列（例如 Group），无法进行分组分析。")
        st.stop()
        
    group_col = st.selectbox("选择分组列 (Group Column)", non_numeric_cols)
    
    unique_groups = raw_df[group_col].dropna().unique()
    if len(unique_groups) < 2:
        st.error("分组列中的组别少于 2 个。")
        st.stop()
        
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        case_group = st.selectbox("实验组 (Case)", unique_groups, index=0)
    with col_sel2:
        control_group = st.selectbox("对照组 (Ctrl)", unique_groups, index=min(1, len(unique_groups)-1))
        
    if case_group == control_group:
        st.warning("⚠️ 实验组和对照组相同，无法分析差异。")
        st.stop()

    st.divider()

    # 3. 参数设置
    st.subheader("3. 统计参数")
    use_log = st.checkbox("执行 Log2 转换", value=True, help="如果上传的数据已经是取过对数的，请取消此项")
    p_thresh = st.number_input("P-value 阈值", value=0.05, step=0.01, format="%.3f")
    fc_thresh = st.number_input("Log2 FC 阈值 (绝对值)", value=1.0, step=0.1)


# ==========================================
# 4. 主逻辑执行
# ==========================================

# A. 预处理
analysis_df, feature_cols = preprocess_data(raw_df, group_col, log_transform=use_log)

# 仅保留选中的两组样本
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
# 5. 结果展示 (Tabs)
# ==========================================

st.title("🧪 代谢组学分析报告")
st.markdown(f"**对比组**: `{case_group}` vs `{control_group}` | **显著特征**: `{len(stats_df[stats_df['Significant'] != 'NS'])}` 个")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 PCA 分析", 
    "🎯 PLS-DA 分析", 
    "🌋 火山图", 
    "📦 详情 (Boxplot)", 
    "📑 数据表"
])

# --- TAB 1: PCA ---
with tab1:
    st.markdown("### 主成分分析 (PCA)")
    col_pca1, col_pca2 = st.columns([3, 1])
    
    with col_pca1:
        X = sub_df[feature_cols]
        # PCA 前必须标准化 (Mean=0, Var=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        
        pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
        pca_df['Group'] = sub_df[group_col].values
        
        var_exp = pca.explained_variance_ratio_
        
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Group',
                             title=f"PCA Score Plot (PC1: {var_exp[0]:.1%}, PC2: {var_exp[1]:.1%})",
                             template="simple_white", width=700, height=500)
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with col_pca2:
        st.info("PCA 是一种无监督分析，用于观察样本的自然聚类情况。如果样本按照组别自然分开，说明组间存在显著的整体代谢差异。")

# --- TAB 2: PLS-DA ---
with tab2:
    st.markdown("### 偏最小二乘判别分析 (PLS-DA)")
    col_pls1, col_pls2 = st.columns([3, 1])
    
    with col_pls1:
        # 准备数据
        X_pls = sub_df[feature_cols]
        y_pls = pd.factorize(sub_df[group_col])[0]
        
        scaler_pls = StandardScaler()
        X_pls_scaled = scaler_pls.fit_transform(X_pls)
        
        # 建立 PLS 模型
        pls = PLSRegression(n_components=2)
        pls.fit(X_pls_scaled, y_pls)
        
        # 1. Score Plot
        pls_scores = pd.DataFrame(pls.x_scores_, columns=['Comp 1', 'Comp 2'])
        pls_scores['Group'] = sub_df[group_col].values
        
        fig_pls = px.scatter(pls_scores, x='Comp 1', y='Comp 2', color='Group',
                             title="PLS-DA Score Plot", template="simple_white")
        st.plotly_chart(fig_pls, use_container_width=True)
        
        # 2. VIP Scores
        st.markdown("#### 变量重要性投影 (VIP Scores)")
        vip_vals = calculate_vips(pls)
        vip_df = pd.DataFrame({'Metabolite': feature_cols, 'VIP': vip_vals})
        vip_df = vip_df.sort_values('VIP', ascending=False).head(15)
        
        fig_vip = px.bar(vip_df, x='VIP', y='Metabolite', orientation='h',
                         color='VIP', title="Top 15 Important Features (VIP)",
                         color_continuous_scale='Teal', template="simple_white")
        fig_vip.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_vip, use_container_width=True)
        
    with col_pls2:
        st.info("PLS-DA 是一种有监督分析，强行寻找能最大化区分组别的方向。\n\n**VIP Score**: 值大于 1 通常被认为对分组贡献显著。")

# --- TAB 3: Volcano Plot ---
with tab3:
    col_vol1, col_vol2 = st.columns([3, 1])
    with col_vol1:
        fig_vol = px.scatter(stats_df, x="Log2_FC", y="-Log10_P", color="Significant",
                             color_discrete_map=color_map,
                             hover_data=["Metabolite", "P_Value", "FDR"],
                             title="Volcano Plot (P-value vs Fold Change)",
                             template="simple_white")
        
        # 辅助线
        fig_vol.add_hline(y=-np.log10(p_thresh), line_dash="dash", line_color="gray")
        fig_vol.add_vline(x=fc_thresh, line_dash="dash", line_color="gray")
        fig_vol.add_vline(x=-fc_thresh, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig_vol, use_container_width=True)
        
    with col_vol2:
        st.write("#### 筛选统计")
        st.metric("上调 (Up)", len(stats_df[stats_df['Significant']=='Up']))
        st.metric("下调 (Down)", len(stats_df[stats_df['Significant']=='Down']))
        st.caption(f"阈值设定: P < {p_thresh}, |Log2FC| > {fc_thresh}")

# --- TAB 4: Box Plot ---
with tab4:
    st.markdown("### 单个代谢物表达水平")
    
    # 优先显示显著差异的代谢物
    sig_feats = stats_df[stats_df['Significant'] != 'NS']['Metabolite'].tolist()
    all_feats = sorted(sub_df[feature_cols].columns.tolist())
    
    # 下拉框
    box_feat = st.selectbox("选择代谢物查看:", sig_feats if sig_feats else all_feats)
    
    if box_feat:
        # 准备画图数据
        plot_data = sub_df[[group_col, box_feat]].copy()
        
        fig_box = px.box(plot_data, x=group_col, y=box_feat, color=group_col,
                         points='all', # 显示散点
                         title=f"Expression of {box_feat}",
                         template="simple_white")
        st.plotly_chart(fig_box, use_container_width=True)

# --- TAB 5: Results Table ---
with tab5:
    st.markdown("### 详细统计结果表")
    
    # 格式化表格用于显示
    out_df = stats_df.sort_values("P_Value").copy()
    
    st.dataframe(
        out_df.style.format({
            "Mean_Case": "{:.2f}", "Mean_Ctrl": "{:.2f}",
            "Log2_FC": "{:.2f}", "P_Value": "{:.4f}", 
            "FDR": "{:.4f}", "-Log10_P": "{:.2f}"
        }).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05),
        use_container_width=True
    )
    
    # 下载按钮
    csv_data = out_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 下载完整结果 (CSV)", data=csv_data, 
                       file_name="metabolomics_results.csv", mime="text/csv")


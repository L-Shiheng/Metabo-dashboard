import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ==========================================
# 0. 全局配置与发表级样式 (Publication Ready)
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro", page_icon="🔬", layout="wide")

# CSS: 调整字体和布局，使其更像专业软件
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 3rem;}
    h1, h2, h3 {font-family: 'Arial', sans-serif;}
    .stAlert {font-weight: bold;}
    /* 调整 Tab 字体大小 */
    button[data-baseweb="tab"] {font-size: 16px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# 定义学术常用的配色方案
COLOR_PALETTE = {
    'Up': '#CD0000',      # Firebrick Red
    'Down': '#008B00',    # Dark Green (或用 #00008B 深蓝)
    'NS': '#D3D3D3'       # Light Grey
}

# Plotly 统一模板函数：SCI 风格 (白底黑框)
def update_layout_pub(fig, title="", x_title="", y_title=""):
    fig.update_layout(
        template="simple_white", # 纯白背景，无网格
        title={
            'text': title,
            'y':0.95, 'x':0.5,
            'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=18, color='black', family="Arial, bold")
        },
        xaxis=dict(title=x_title, showline=True, linewidth=1.5, linecolor='black', mirror=True, title_font=dict(size=16)),
        yaxis=dict(title=y_title, showline=True, linewidth=1.5, linecolor='black', mirror=True, title_font=dict(size=16)),
        font=dict(family="Arial", size=14, color="black"),
        width=700, height=550,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            bordercolor="Black", borderwidth=1
        )
    )
    return fig

# ==========================================
# 1. 核心计算函数
# ==========================================

@st.cache_data
def preprocess_data(df, group_col, log_transform=True):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols: numeric_cols.remove(group_col)
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    # 简单的缺失值填充 (最小值的一半)
    if data_df.isnull().sum().sum() > 0:
        data_df.fillna(data_df.min().min() * 0.5, inplace=True)
    if log_transform:
        data_df = np.log2(data_df + 1)
        
    return pd.concat([df[meta_cols], data_df], axis=1), numeric_cols

def calculate_vips(model):
    """计算 PLS-DA VIP 值"""
    t = model.x_scores_; w = model.x_weights_; q = model.y_loadings_
    p, h = w.shape; vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips

@st.cache_data
def run_statistics(df, group_col, case, control, features):
    g1 = df[df[group_col] == case]
    g2 = df[df[group_col] == control]
    res = []
    for f in features:
        v1, v2 = g1[f].values, g2[f].values
        # Fold Change
        fc = np.mean(v1) - np.mean(v2)
        # T-test
        try: t, p = stats.ttest_ind(v1, v2, equal_var=False)
        except: p = 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p})
    
    res_df = pd.DataFrame(res)
    res_df = res_df.dropna()
    # FDR Correction
    if not res_df.empty:
        _, p_corr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
        res_df['FDR'] = p_corr
        res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
    else:
        res_df['FDR'] = 1.0; res_df['-Log10_P'] = 0
        
    return res_df

# ==========================================
# 2. 界面与主逻辑
# ==========================================
with st.sidebar:
    st.title("🧪 MetaboAnalyst Pro")
    st.markdown("---")
    uploaded_file = st.file_uploader("1. 上传 CSV 数据", type=["csv"])
    
    if not uploaded_file:
        st.info("请上传 CSV。格式：行(样本) x 列(代谢物)。需包含分组列。")
        st.stop()
        
    raw_df = pd.read_csv(uploaded_file)
    non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not non_num: 
        st.error("数据中缺少非数值的分组列！"); st.stop()
    
    group_col = st.selectbox("2. 选择分组列", non_num)
    grps = raw_df[group_col].unique()
    if len(grps) < 2: 
        st.error("分组数量少于2个！"); st.stop()
    
    c1, c2 = st.columns(2)
    case = c1.selectbox("Case (Exp)", grps, index=0)
    ctrl = c2.selectbox("Control", grps, index=min(1, len(grps)-1))
    
    st.markdown("---")
    st.markdown("### ⚙️ 统计阈值")
    p_th = st.number_input("P-value Cutoff", 0.05, format="%.3f")
    fc_th = st.number_input("Log2 FC Cutoff", 1.0)
    
# --- 数据处理流程 ---
df_proc, feats = preprocess_data(raw_df, group_col)
# 筛选两组数据
df_sub = df_proc[df_proc[group_col].isin([case, ctrl])].copy()
# 运行统计
res_stats = run_statistics(df_sub, group_col, case, ctrl, feats)

# 标记显著性
res_stats['Sig'] = 'NS'
res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] > fc_th), 'Sig'] = 'Up'
res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] < -fc_th), 'Sig'] = 'Down'

# 提取显著特征
sig_metabolites = res_stats[res_stats['Sig'] != 'NS']['Metabolite'].tolist()

# ==========================================
# 3. 结果展示 Tab 页
# ==========================================
st.header(f"📊 分析报告: {case} vs {ctrl}")
st.markdown(f"**显著差异代谢物数量**: `{len(sig_metabolites)}` (Up: `{len(res_stats[res_stats['Sig']=='Up'])}`, Down: `{len(res_stats[res_stats['Sig']=='Down'])}`)")

tab1, tab2, tab3, tab4 = st.tabs(["📊 多变量分析 (PCA/PLS)", "🌋 差异火山图", "🔥 聚类热图", "📑 详细结果与箱线图"])

# --- Tab 1: PCA & PLS-DA ---
with tab1:
    col1, col2 = st.columns(2)
    # 准备数据矩阵 (标准化)
    X = StandardScaler().fit_transform(df_sub[feats])
    
    # 1. PCA Plot
    with col1:
        pca = PCA(n_components=2).fit(X)
        pcs = pca.transform(X)
        var = pca.explained_variance_ratio_
        
        fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=df_sub[group_col],
                             width=600, height=500)
        fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black')))
        update_layout_pub(fig_pca, "PCA Score Plot", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
        st.plotly_chart(fig_pca, use_container_width=True)

    # 2. PLS-DA Plot
    with col2:
        pls = PLSRegression(n_components=2).fit(X, pd.factorize(df_sub[group_col])[0])
        pls_scores = pls.x_scores_
        
        fig_pls = px.scatter(x=pls_scores[:,0], y=pls_scores[:,1], color=df_sub[group_col],
                             width=600, height=500)
        fig_pls.update_traces(marker=dict(size=14, symbol='diamond', line=dict(width=1, color='black')))
        update_layout_pub(fig_pls, "PLS-DA Score Plot", "Component 1", "Component 2")
        st.plotly_chart(fig_pls, use_container_width=True)
        
    st.info("💡 提示：将鼠标悬停在图表右上角，点击相机图标即可下载高清图片。")

# --- Tab 2: 火山图 ---
with tab2:
    color_map = {'Up': COLOR_PALETTE['Up'], 'Down': COLOR_PALETTE['Down'], 'NS': COLOR_PALETTE['NS']}
    
    fig_vol = px.scatter(res_stats, x="Log2_FC", y="-Log10_P", color="Sig",
                         color_discrete_map=color_map,
                         hover_data=["Metabolite", "P_Value", "FDR"],
                         width=800, height=600)
    
    # 阈值辅助线
    fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="black", opacity=0.6)
    fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="black", opacity=0.6)
    fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="black", opacity=0.6)
    
    fig_vol.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='black')))
    update_layout_pub(fig_vol, "Volcano Plot", "Log2 Fold Change", "-Log10(P-value)")
    
    col_v1, col_v2 = st.columns([3, 1])
    with col_v1:
        st.plotly_chart(fig_vol, use_container_width=True)
    with col_v2:
        st.markdown("#### 图例说明")
        st.markdown(f"🔴 **Up**: P < {p_th} & FC > {fc_th}")
        st.markdown(f"🟢 **Down**: P < {p_th} & FC < -{fc_th}")
        st.markdown("⚪ **NS**: Not Significant")

# --- Tab 3: 聚类热图 ---
with tab3:
    st.subheader("Top 30 显著差异代谢物热图")
    
    if len(sig_metabolites) < 2:
        st.warning(f"显著差异代谢物不足 2 个 (当前: {len(sig_metabolites)})，无法绘制聚类热图。请尝试调大 P-value 阈值。")
    else:
        # 取最显著的前30个
        top_n = 30
        top_feats = res_stats.sort_values('P_Value').head(top_n)['Metabolite'].tolist()
        hm_data = df_sub.set_index(group_col)[top_feats]
        
        # 颜色条
        lut = dict(zip(df_sub[group_col].unique(), "rbg"))
        row_colors = df_sub[group_col].map(lut)
        
        try:
            # 绘图
            g = sns.clustermap(hm_data.astype(float), 
                               z_score=1,  # 按列标准化
                               cmap="vlag", # 蓝-白-红 学术配色
                               center=0, 
                               row_colors=row_colors,
                               figsize=(10, 10),
                               dendrogram_ratio=(.15, .15),
                               cbar_pos=(.02, .8, .03, .15)) # 调整 colorbar 位置
            
            g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=45, ha="right", fontsize=9)
            g.ax_heatmap.set_yticklabels([]) # 隐藏样本名称
            g.ax_heatmap.set_ylabel("Samples", fontsize=12)
            
            # 传递 Figure 对象给 Streamlit
            st.pyplot(g.fig)
            
        except Exception as e:
            st.error(f"热图绘制失败 (数据可能含有太多 NaN 或 方差为 0): {e}")

# --- Tab 4: 结果表 & 箱线图 ---
with tab4:
    col_d1, col_d2 = st.columns([1.5, 1])
    
    with col_d1:
        st.subheader("📑 统计结果表")
        # 格式化表格
        display_df = res_stats.sort_values("P_Value").copy()
        st.dataframe(
            display_df.style.format({
                "Log2_FC": "{:.2f}", "P_Value": "{:.4f}", "FDR": "{:.4f}", "-Log10_P": "{:.2f}"
            }).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05),
            use_container_width=True, height=500
        )
        # 下载
        csv = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 下载完整结果 (CSV)", csv, "metabo_results.csv", "text/csv")
        
    with col_d2:
        st.subheader("📦 单个代谢物详情")
        # 联动：选择代谢物
        all_options = sorted(res_stats['Metabolite'].tolist())
        # 默认选最显著的那个
        default_idx = all_options.index(display_df.iloc[0]['Metabolite']) if not display_df.empty else 0
        target_feat = st.selectbox("选择代谢物查看箱线图:", all_options, index=default_idx)
        
        if target_feat:
            # 绘制箱线图
            box_df = df_sub[[group_col, target_feat]].copy()
            fig_box = px.box(box_df, x=group_col, y=target_feat, color=group_col,
                             points="all", # 显示所有散点
                             width=400, height=450)
            
            fig_box.update_traces(marker=dict(size=6, opacity=0.7))
            update_layout_pub(fig_box, f"{target_feat}", "Group", "Log2 Intensity")
            st.plotly_chart(fig_box, use_container_width=True)


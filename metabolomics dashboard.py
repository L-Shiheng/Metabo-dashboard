import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

# ==========================================
# 0. 导入数据清洗模块
# ==========================================
try:
    from data_preprocessing import data_cleaning_pipeline
except ImportError:
    st.error("❌ 错误：未找到 'data_preprocessing.py'。请确保该文件在同一目录下。")
    st.stop()

# ==========================================
# 1. 全局配置
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro (Multi-Group)", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3 {font-family: 'Arial', sans-serif; color: #2c3e50;}
    button[data-baseweb="tab"] {font-size: 16px; font-weight: bold; padding: 10px 15px;}
    .stMultiSelect [data-baseweb="tag"] {background-color: #f0f2f6;}
</style>
""", unsafe_allow_html=True)

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'}
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

# 通用绘图布局 (修复图例遮挡问题)
def update_layout_square(fig, title="", x_title="", y_title="", width=600, height=600):
    fig.update_layout(
        template="simple_white",
        width=width, height=height,
        title={
            'text': title,
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=20, color='black', family="Arial, bold")
        },
        xaxis=dict(
            title=x_title, showline=True, linewidth=2, linecolor='black', mirror=True, 
            title_font=dict(size=16, family="Arial, bold"),
            tickfont=dict(size=14, family="Arial")
        ),
        yaxis=dict(
            title=y_title, showline=True, linewidth=2, linecolor='black', mirror=True, 
            title_font=dict(size=16, family="Arial, bold"),
            tickfont=dict(size=14, family="Arial"),
            automargin=True
        ),
        legend=dict(
            yanchor="top", y=1,      # 图例顶部对齐
            xanchor="left", x=1.02,  # 关键修改：x > 1 把图例移到框外
            bordercolor="Black", borderwidth=0, # 去掉边框更清爽
            bgcolor="rgba(255,255,255,0)" # 透明背景
        ),
        margin=dict(l=80, r=120, t=80, b=80) # 增加右边距(r)以容纳图例
    )
    return fig

# PLS-DA 椭圆辅助函数
def get_ellipse_coordinates(x, y, std_mult=2):
    if len(x) < 3: return None, None
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * std_mult * np.sqrt(vals)
    t = np.linspace(0, 2*np.pi, 100)
    ell_x = width/2 * np.cos(t)
    ell_y = height/2 * np.sin(t)
    rad = np.radians(theta)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    ell_coords = np.dot(R, np.array([ell_x, ell_y]))
    return ell_coords[0] + mean_x, ell_coords[1] + mean_y

# VIP 计算
def calculate_vips(model):
    t = model.x_scores_; w = model.x_weights_; q = model.y_loadings_
    p, h = w.shape; vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    return vips

# 统计分析逻辑
@st.cache_data
def run_pairwise_statistics(df, group_col, case, control, features):
    g1 = df[df[group_col] == case]
    g2 = df[df[group_col] == control]
    res = []
    for f in features:
        v1, v2 = g1[f].values, g2[f].values
        fc = np.mean(v1) - np.mean(v2)
        try: t, p = stats.ttest_ind(v1, v2, equal_var=False)
        except: p = 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p})
    
    res_df = pd.DataFrame(res)
    res_df = res_df.dropna()
    if not res_df.empty:
        _, p_corr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
        res_df['FDR'] = p_corr
        res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
    else:
        res_df['FDR'] = 1.0; res_df['-Log10_P'] = 0
    return res_df

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.header("🛠️ 设置面板")
    uploaded_file = st.file_uploader("1. 上传 CSV 数据", type=["csv"])
    
    if not uploaded_file:
        st.info("请上传数据以开始。")
        st.stop()
        
    raw_df = pd.read_csv(uploaded_file)
    non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not non_num: st.error("错误：未找到分组列"); st.stop()
    group_col = st.selectbox("2. 分组列", non_num)
    
    all_groups = sorted(raw_df[group_col].unique())
    
    st.markdown("### 3. 选择分析组别")
    selected_groups = st.multiselect(
        "选择要包含在分析中的组 (至少2个):",
        all_groups,
        default=all_groups[:2] if len(all_groups) >= 2 else all_groups
    )
    
    if len(selected_groups) < 2:
        st.error("请至少选择两个组进行分析。")
        st.stop()
        
    st.divider()
    
    st.markdown("### 4. 差异比较设置 (火山图)")
    c1, c2 = st.columns(2)
    # 过滤出已选中的组供选择
    valid_groups = [g for g in selected_groups]
    case_grp = c1.selectbox("Exp (Case)", valid_groups, index=0)
    # 智能默认值
    default_ctrl_idx = 1 if len(valid_groups) > 1 else 0
    ctrl_grp = c2.selectbox("Ctrl (Ref)", valid_groups, index=default_ctrl_idx)
    
    if case_grp == ctrl_grp:
        st.warning("⚠️ Case 和 Control 相同。")

    st.divider()
    st.subheader("5. 统计阈值")
    p_th = st.number_input("P-value", 0.05, format="%.3f")
    fc_th = st.number_input("Log2 FC", 1.0)
    
    # 新增：控制抖动 (Jitter)
    st.divider()
    st.markdown("### 6. 图表微调")
    enable_jitter = st.checkbox("开启火山图抖动 (Jitter)", value=True, help="如果火山图呈现奇怪的线条状，请开启此选项以分散点。")

# ==========================================
# 3. 主逻辑
# ==========================================
df_proc, feats = data_cleaning_pipeline(raw_df, group_col, impute_na=True, log_transform=True)
df_sub = df_proc[df_proc[group_col].isin(selected_groups)].copy()

if case_grp != ctrl_grp:
    res_stats = run_pairwise_statistics(df_sub, group_col, case_grp, ctrl_grp, feats)
    
    res_stats['Sig'] = 'NS'
    res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] > fc_th), 'Sig'] = 'Up'
    res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] < -fc_th), 'Sig'] = 'Down'
    sig_metabolites = res_stats[res_stats['Sig'] != 'NS']['Metabolite'].tolist()
else:
    res_stats = pd.DataFrame()
    sig_metabolites = []

# ==========================================
# 4. 结果展示
# ==========================================
st.title("📊 代谢组学分析报告")
st.markdown(f"**当前概览组别**: {', '.join(selected_groups)}")
st.markdown(f"**当前差异对比**: `{case_grp}` vs `{ctrl_grp}`")

tabs = st.tabs(["📊 PCA", "🎯 PLS-DA", "⭐ VIP 特征", "🌋 火山图", "🔥 热图", "📑 详情"])

# --- Tab 1: PCA ---
with tabs[0]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if len(df_sub) < 3: st.warning("样本量过少，无法进行 PCA。")
        else:
            X = StandardScaler().fit_transform(df_sub[feats])
            pca = PCA(n_components=2).fit(X)
            pcs = pca.transform(X)
            var = pca.explained_variance_ratio_
            
            fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=df_sub[group_col], symbol=df_sub[group_col],
                                 color_discrete_sequence=GROUP_COLORS, width=600, height=600)
            fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
            update_layout_square(fig_pca, "PCA Score Plot", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
            st.plotly_chart(fig_pca, use_container_width=False)

# --- Tab 2: PLS-DA ---
with tabs[1]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if len(df_sub) < 3: st.warning("样本量过少。")
        else:
            X_pls = StandardScaler().fit_transform(df_sub[feats])
            y_labels = pd.factorize(df_sub[group_col])[0]
            pls_model = PLSRegression(n_components=2).fit(X_pls, y_labels)
            pls_scores = pls_model.x_scores_
            plot_df = pd.DataFrame({'C1': pls_scores[:,0], 'C2': pls_scores[:,1], 'Group': df_sub[group_col].values})
            
            fig_pls = px.scatter(plot_df, x='C1', y='C2', color='Group', symbol='Group',
                                 color_discrete_sequence=GROUP_COLORS, width=600, height=600)
            
            for i, grp in enumerate(selected_groups):
                sub_g = plot_df[plot_df['Group'] == grp]
                if len(sub_g) >= 3:
                    ell_x, ell_y = get_ellipse_coordinates(sub_g['C1'], sub_g['C2'])
                    if ell_x is not None:
                        color = GROUP_COLORS[i % len(GROUP_COLORS)]
                        fig_pls.add_trace(go.Scatter(x=ell_x, y=ell_y, mode='lines', line=dict(color=color, width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
            
            fig_pls.update_traces(marker=dict(size=15, line=dict(width=1.5, color='black'), opacity=1.0))
            update_layout_square(fig_pls, "PLS-DA Score Plot", "Component 1", "Component 2")
            st.plotly_chart(fig_pls, use_container_width=False)

# --- Tab 3: VIP ---
with tabs[2]:
    st.markdown("### Top 25 VIP Features")
    if 'pls_model' in locals():
        vip_scores = calculate_vips(pls_model)
        vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': vip_scores})
        top_vip = vip_df.sort_values('VIP', ascending=True).tail(25)
        
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            fig_vip = px.bar(top_vip, x="VIP", y="Metabolite", orientation='h',
                             color="VIP", color_continuous_scale="RdBu_r", width=800, height=700)
            fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
            fig_vip.update_traces(marker_line_color='black', marker_line_width=1.0)
            
            # 使用 update_layout 手动调整，不强制正方形
            fig_vip.update_layout(
                template="simple_white", width=800, height=700,
                title={'text': "VIP Scores", 'x':0.5, 'xanchor': 'center', 'font': dict(size=20, family="Arial, bold")},
                xaxis=dict(title="VIP Score", showline=True, mirror=True, linewidth=2, linecolor='black'),
                yaxis=dict(title="", showline=True, mirror=True, linewidth=2, linecolor='black', automargin=True),
                coloraxis_showscale=False,
                margin=dict(l=200, r=40, t=80, b=80) 
            )
            st.plotly_chart(fig_vip, use_container_width=False)

# --- Tab 4: 火山图 (修复曲线问题) ---
with tabs[3]:
    if case_grp == ctrl_grp:
        st.warning("⚠️ 请选择不同的组。")
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # 准备绘图数据
            plot_df = res_stats.copy()
            
            x_col = "Log2_FC"
            y_col = "-Log10_P"
            
            # --- 关键修复：加入随机抖动 (Jitter) ---
            if enable_jitter:
                # 生成微小的随机噪音，比例为数据范围的 1%
                np.random.seed(42) # 保证每次抖动一致
                x_range = plot_df['Log2_FC'].max() - plot_df['Log2_FC'].min()
                y_range = plot_df['-Log10_P'].max() - plot_df['-Log10_P'].min()
                
                # 如果范围为0 (所有值都一样)，给一个默认范围防止报错
                if x_range == 0: x_range = 1
                if y_range == 0: y_range = 1
                
                plot_df['Log2_FC_Jitter'] = plot_df['Log2_FC'] + np.random.normal(0, x_range*0.015, len(plot_df))
                plot_df['-Log10_P_Jitter'] = plot_df['-Log10_P'] + np.random.normal(0, y_range*0.015, len(plot_df))
                
                x_col = "Log2_FC_Jitter"
                y_col = "-Log10_P_Jitter"
            
            fig_vol = px.scatter(plot_df, x=x_col, y=y_col, color="Sig",
                                 color_discrete_map=COLOR_PALETTE,
                                 # 即使加了抖动，Hover依然显示真实的数值
                                 hover_data={"Metabolite":True, "Log2_FC":':.2f', "P_Value":':.2e', x_col:False, y_col:False},
                                 width=600, height=600)
            
            fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="black", opacity=0.8)
            fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="black", opacity=0.8)
            
            fig_vol.update_traces(marker=dict(size=10, opacity=0.7, line=dict(width=1, color='black')))
            update_layout_square(fig_vol, f"Volcano: {case_grp} vs {ctrl_grp}", "Log2 Fold Change", "-Log10(P-value)")
            st.plotly_chart(fig_vol, use_container_width=False)

# --- Tab 5: 热图 ---
with tabs[4]:
    if not sig_metabolites: st.warning("无显著差异物。")
    else:
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            top_n = 50
            top_feats = res_stats.sort_values('P_Value').head(top_n)['Metabolite'].tolist()
            hm_data = df_sub.set_index(group_col)[top_feats]
            lut = {grp: GROUP_COLORS[i % len(GROUP_COLORS)] for i, grp in enumerate(df_sub[group_col].unique())}
            row_colors = df_sub[group_col].map(lut)
            
            try:
                g = sns.clustermap(hm_data.astype(float), z_score=1, cmap="vlag", center=0, 
                                   row_colors=row_colors, figsize=(10, 10), 
                                   dendrogram_ratio=(.15, .15), cbar_pos=(.02, .8, .03, .12))
                g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=45, ha="right", fontsize=10)
                g.ax_heatmap.set_yticklabels([]); g.ax_heatmap.set_ylabel("Samples", fontsize=12)
                st.pyplot(g.fig)
            except Exception as e: st.error(f"Error: {e}")

# --- Tab 6: 详情 ---
with tabs[5]:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("统计表")
        if not res_stats.empty:
            display_df = res_stats.sort_values("P_Value").copy()
            st.dataframe(display_df.style.format({"Log2_FC": "{:.2f}", "P_Value": "{:.2e}", "FDR": "{:.2e}"})
                         .background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05),
                         use_container_width=True, height=600)
    with c2:
        st.subheader("箱线图")
        feat_options = sorted(feats)
        default_ix = feat_options.index(sig_metabolites[0]) if sig_metabolites else 0
        target_feat = st.selectbox("选择代谢物", feat_options, index=default_ix)
        
        if target_feat:
            box_df = df_sub[[group_col, target_feat]].copy()
            fig_box = px.box(box_df, x=group_col, y=target_feat, color=group_col,
                             color_discrete_sequence=GROUP_COLORS, points="all", width=500, height=500)
            fig_box.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='black')), jitter=0.5, pointpos=0)
            update_layout_square(fig_box, target_feat, "Group", "Log2 Intensity", width=500, height=500)
            st.plotly_chart(fig_box, use_container_width=False)


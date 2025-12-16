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

# 导入您刚刚写的外部模块
try:
    from data_preprocessing import data_cleaning_pipeline
except ImportError:
    st.error("请确保 'data_preprocessing.py' 文件在同一目录下！")
    st.stop()

# ==========================================
# 0. 全局配置与样式
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 3rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3 {font-family: 'Arial', sans-serif; color: #2c3e50;}
    button[data-baseweb="tab"] {font-size: 16px; font-weight: bold; padding: 10px 20px;}
</style>
""", unsafe_allow_html=True)

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'}
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488']

# 通用绘图布局函数 (正方形优化)
def update_layout_square(fig, title="", x_title="", y_title="", width=600, height=600):
    fig.update_layout(
        template="simple_white",
        width=width, height=height, # 强制宽高相等
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
            automargin=True # 自动调整边距防止标签被切
        ),
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            bordercolor="Black", borderwidth=1,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(l=80, r=40, t=80, b=80)
    )
    return fig

# PLS-DA 椭圆计算函数
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
def run_statistics(df, group_col, case, control, features):
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
# 1. 侧边栏与数据导入
# ==========================================
with st.sidebar:
    st.header("🛠️ 设置面板")
    uploaded_file = st.file_uploader("1. 上传数据 (CSV)", type=["csv"])
    
    if not uploaded_file:
        st.info("请先上传数据 CSV")
        st.stop()
        
    raw_df = pd.read_csv(uploaded_file)
    non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not non_num: st.error("错误：没有找到分组列"); st.stop()
    
    group_col = st.selectbox("2. 分组列", non_num)
    grps = raw_df[group_col].unique()
    if len(grps) < 2: st.error("错误：组别少于2个"); st.stop()
    
    c1, c2 = st.columns(2)
    case = c1.selectbox("Exp (Case)", grps, index=0)
    ctrl = c2.selectbox("Ctrl", grps, index=min(1, len(grps)-1))
    
    st.divider()
    st.subheader("3. 统计阈值")
    p_th = st.number_input("P-value", 0.05, format="%.3f")
    fc_th = st.number_input("Log2 FC", 1.0)

# ==========================================
# 2. 数据处理与计算 (调用外部模块)
# ==========================================
# 调用 data_preprocessing.py 中的函数
df_proc, feats = data_cleaning_pipeline(raw_df, group_col, impute_na=True, log_transform=True)

df_sub = df_proc[df_proc[group_col].isin([case, ctrl])].copy()
res_stats = run_statistics(df_sub, group_col, case, ctrl, feats)

res_stats['Sig'] = 'NS'
res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] > fc_th), 'Sig'] = 'Up'
res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] < -fc_th), 'Sig'] = 'Down'
sig_metabolites = res_stats[res_stats['Sig'] != 'NS']['Metabolite'].tolist()

# ==========================================
# 3. 结果展示
# ==========================================
st.title(f"📊 分析报告: {case} vs {ctrl}")

tabs = st.tabs(["📊 PCA 分析", "🎯 PLS-DA 分析", "🌋 差异火山图", "🔥 聚类热图", "📑 结果与箱线图"])

# --- Tab 1: PCA ---
with tabs[0]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        X = StandardScaler().fit_transform(df_sub[feats])
        pca = PCA(n_components=2).fit(X)
        pcs = pca.transform(X)
        var = pca.explained_variance_ratio_
        
        fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=df_sub[group_col],
                             symbol=df_sub[group_col], 
                             color_discrete_sequence=GROUP_COLORS,
                             width=600, height=600)
        fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
        update_layout_square(fig_pca, "PCA Score Plot", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
        st.plotly_chart(fig_pca, use_container_width=False)

# --- Tab 2: PLS-DA & VIP ---
with tabs[1]:
    col1, col2 = st.columns(2)
    
    X_pls = StandardScaler().fit_transform(df_sub[feats])
    y_labels = pd.factorize(df_sub[group_col])[0]
    pls_model = PLSRegression(n_components=2)
    pls_model.fit(X_pls, y_labels)
    pls_scores = pls_model.x_scores_
    plot_df = pd.DataFrame({'C1': pls_scores[:,0], 'C2': pls_scores[:,1], 'Group': df_sub[group_col].values})
    
    with col1:
        st.markdown("#### 1. PLS-DA Score Plot")
        fig_pls = px.scatter(plot_df, x='C1', y='C2', color='Group', symbol='Group',
                             color_discrete_sequence=GROUP_COLORS,
                             width=600, height=600)
        
        for i, grp in enumerate(df_sub[group_col].unique()):
            sub_g = plot_df[plot_df['Group'] == grp]
            ell_x, ell_y = get_ellipse_coordinates(sub_g['C1'], sub_g['C2'])
            if ell_x is not None:
                color = GROUP_COLORS[i % len(GROUP_COLORS)]
                fig_pls.add_trace(go.Scatter(x=ell_x, y=ell_y, mode='lines', 
                                             line=dict(color=color, width=2, dash='dash'),
                                             showlegend=False, hoverinfo='skip'))
        fig_pls.update_traces(marker=dict(size=15, line=dict(width=1.5, color='black'), opacity=1.0))
        update_layout_square(fig_pls, "PLS-DA Score Plot", "Component 1", "Component 2")
        st.plotly_chart(fig_pls, use_container_width=False)
        
    with col2:
        st.markdown("#### 2. VIP Scores (Top 25)")
        vip_scores = calculate_vips(pls_model)
        vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': vip_scores})
        top_vip = vip_df.sort_values('VIP', ascending=True).tail(25)
        
        # 修复：设置为正方形 600x600，且通过 bar chart 清晰展示
        fig_vip = px.bar(top_vip, x="VIP", y="Metabolite", orientation='h',
                         color="VIP", color_continuous_scale="RdBu_r",
                         width=600, height=600) # 这里改为 600x600 正方形

        fig_vip.add_vline(x=1.0, line_dash="dash", line_color="black")
        fig_vip.update_traces(marker_line_color='black', marker_line_width=1.0)
        
        # 使用 update_layout_square 保持风格一致，但因为是条形图，可能需要微调
        fig_vip.update_layout(
            template="simple_white",
            width=600, height=600, # 强制正方形
            title={'text': "VIP Scores (Top 25)", 'x':0.5, 'xanchor': 'center', 'font': dict(size=20, family="Arial, bold")},
            xaxis=dict(title="VIP Score", showline=True, mirror=True, linewidth=2, linecolor='black'),
            yaxis=dict(title="", showline=True, mirror=True, linewidth=2, linecolor='black'),
            margin=dict(l=10, r=20, t=60, b=60),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_vip, use_container_width=False)

# --- Tab 3: 火山图 ---
with tabs[2]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # 修复 1: 鼠标悬停显示代谢物名字 (hover_data)
        fig_vol = px.scatter(res_stats, x="Log2_FC", y="-Log10_P", color="Sig",
                             color_discrete_map=COLOR_PALETTE,
                             hover_data=["Metabolite", "P_Value", "FDR"], # 加上了名字
                             width=600, height=600)
        
        # 修复 2: 统计学竖线 (Threshold Lines)
        # P值线
        fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="black", opacity=0.8)
        # FC线 (Log2 FC 阈值)
        fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="black", opacity=0.8)
        fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="black", opacity=0.8)
        
        fig_vol.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='black')))
        update_layout_square(fig_vol, "Volcano Plot", "Log2 Fold Change", "-Log10(P-value)")
        st.plotly_chart(fig_vol, use_container_width=False)

# --- Tab 4: 聚类热图 ---
with tabs[3]:
    if len(sig_metabolites) < 2:
        st.warning("显著差异物太少，无法绘制热图。")
    else:
        c1, c2, c3 = st.columns([1, 6, 1])
        with c2:
            top_n = 40
            top_feats = res_stats.sort_values('P_Value').head(top_n)['Metabolite'].tolist()
            hm_data = df_sub.set_index(group_col)[top_feats]
            lut = dict(zip(df_sub[group_col].unique(), "rbg"))
            row_colors = df_sub[group_col].map(lut)
            try:
                g = sns.clustermap(hm_data.astype(float), z_score=1, cmap="vlag", center=0, 
                                   row_colors=row_colors, figsize=(10, 10), 
                                   dendrogram_ratio=(.15, .15), cbar_pos=(.02, .8, .03, .12))
                g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), rotation=45, ha="right", fontsize=10)
                g.ax_heatmap.set_yticklabels([])
                g.ax_heatmap.set_ylabel("Samples", fontsize=12)
                st.pyplot(g.fig)
            except Exception as e: st.error(f"绘图错误: {e}")

# --- Tab 5: 结果与箱线图 ---
with tabs[4]:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("详细数据表")
        display_df = res_stats.sort_values("P_Value").copy()
        st.dataframe(display_df.style.format({"Log2_FC": "{:.2f}", "P_Value": "{:.2e}", "FDR": "{:.2e}"})
                     .background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05),
                     use_container_width=True, height=600)
        
    with c2:
        st.subheader("单变量箱线图")
        target_feat = st.selectbox("选择代谢物", sorted(res_stats['Metabolite'].tolist()))
        if target_feat:
            box_df = df_sub[[group_col, target_feat]].copy()
            # 修复 3: 箱线图圆点问题
            # points="all" 默认是放在箱子旁边的。
            # 要想放在箱子里面/上面，需要调整 pointpos=0
            fig_box = px.box(box_df, x=group_col, y=target_feat, color=group_col,
                             color_discrete_sequence=GROUP_COLORS,
                             points="all", # 显示所有点
                             width=500, height=500)
            
            # 关键修改：pointpos=0 把点对齐到中心，jitter 控制抖动宽度
            fig_box.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='black')),
                                  jitter=0.5, pointpos=0) 
            
            update_layout_square(fig_box, target_feat, "Group", "Log2 Intensity", width=500, height=500)
            st.plotly_chart(fig_box, use_container_width=False)

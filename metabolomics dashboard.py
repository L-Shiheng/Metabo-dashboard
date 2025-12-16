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
# 0. 全局配置与风格设置 (Publication Ready)
# ==========================================
st.set_page_config(page_title="MetaboAnalyst Pro", page_icon="🔬", layout="wide")

# CSS: 调整字体和布局，使其更像专业软件
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 3rem;}
    h1, h2, h3 {font-family: 'Arial', sans-serif;}
    .stAlert {font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# 定义学术常用的配色方案 (MetaboAnalyst 风格: 红/绿 或 红/蓝)
COLOR_PALETTE = {
    'Up': '#CD0000',      # 深红
    'Down': '#008B00',    # 深绿 (或改为 '#00008B' 深蓝)
    'NS': '#D3D3D3'       # 浅灰
}

# Plotly 统一模板函数：让所有交互图看起来像打印出来的文章插图
def update_layout_pub(fig, title="", x_title="", y_title=""):
    fig.update_layout(
        template="simple_white", # 纯白背景，无网格
        title={
            'text': title,
            'y':0.95, 'x':0.5,
            'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=18, color='black', family="Arial, bold")
        },
        xaxis=dict(title=x_title, showline=True, linewidth=1.5, linecolor='black', mirror=True),
        yaxis=dict(title=y_title, showline=True, linewidth=1.5, linecolor='black', mirror=True),
        font=dict(family="Arial", size=14, color="black"),
        width=800, height=600,
        margin=dict(l=60, r=40, t=60, b=60)
    )
    return fig

# ==========================================
# 1. 核心计算函数 (含通路数据库)
# ==========================================

# --- 内置微型通路数据库 (仅作演示，真实分析需连接 KEGG API) ---
PATHWAY_DB = {
    "Glycolysis / Gluconeogenesis": ["Glucose", "Pyruvate", "Lactate", "Hexokinase", "Fructose-6P", "G3P"],
    "Citrate Cycle (TCA cycle)": ["Citrate", "Succinate", "Fumarate", "Malate", "Oxaloacetate", "Pyruvate", "Acetyl-CoA"],
    "Pyruvate Metabolism": ["Pyruvate", "Lactate", "Acetyl-CoA", "Acetate", "Acetaldehyde"],
    "Alanine, Aspartate and Glutamate": ["Alanine", "Aspartate", "Glutamate", "Glutamine", "Pyruvate", "Oxaloacetate"],
    "Glycerolipid Metabolism": ["Glycerol", "Triglyceride", "G3P", "Fatty Acid"],
    "Fatty Acid Biosynthesis": ["Acetyl-CoA", "Malonyl-CoA", "Fatty Acid", "Pyruvate"]
}

@st.cache_data
def run_pathway_analysis(significant_metabolites, all_metabolites_in_study):
    """
    执行简易的通路富集分析 (Fisher Exact Test / Hypergeometric Test)
    """
    results = []
    # 简单的模糊匹配：只要列名里包含关键字就算匹配
    sig_set = set([m.lower() for m in significant_metabolites])
    bg_set = set([m.lower() for m in all_metabolites_in_study])
    
    for pathway_name, compounds in PATHWAY_DB.items():
        path_set = set([c.lower() for c in compounds])
        
        # a: 既在通路里，又显著的 (Hit)
        hits = sig_set.intersection(path_set)
        a = len(hits)
        
        # b: 在通路里，但不显著
        b = len(path_set) - a
        
        # c: 不在通路里，但显著
        c = len(sig_set) - a
        
        # d: 既不在通路里，也不显著 (背景噪音)
        # 估算总背景库大小，这里假设一个常见的人类代谢物库大小为 300
        total_genome = 300 
        d = total_genome - a - b - c
        
        if a > 0: # 只有命中的通路才计算
            oddsratio, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
            results.append({
                'Pathway': pathway_name,
                'Hits': a,
                'P_Value': pvalue,
                '-Log10_P': -np.log10(pvalue) if pvalue > 0 else 0,
                'Impact': a / len(path_set) # 简易 Impact 计算
            })
            
    return pd.DataFrame(results)

@st.cache_data
def preprocess_data(df, group_col, log_transform=True):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols: numeric_cols.remove(group_col)
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    if data_df.isnull().sum().sum() > 0:
        data_df.fillna(data_df.min().min() * 0.5, inplace=True)
    if log_transform:
        data_df = np.log2(data_df + 1)
        
    return pd.concat([df[meta_cols], data_df], axis=1), numeric_cols

def calculate_vips(model):
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
        fc = np.mean(v1) - np.mean(v2)
        try: t, p = stats.ttest_ind(v1, v2, equal_var=False)
        except: p = 1.0
        res.append({'Metabolite': f, 'Log2_FC': fc, 'P_Value': p})
    
    res_df = pd.DataFrame(res)
    res_df = res_df.dropna()
    _, p_corr, _, _ = multipletests(res_df['P_Value'], method='fdr_bh')
    res_df['FDR'] = p_corr
    res_df['-Log10_P'] = -np.log10(res_df['P_Value'])
    return res_df

# ==========================================
# 2. 界面逻辑
# ==========================================
with st.sidebar:
    st.title("🧪 MetaboAnalyst Pro")
    uploaded_file = st.file_uploader("上传 CSV 数据", type=["csv"])
    if not uploaded_file:
        st.info("请上传 CSV。格式：行(样本) x 列(代谢物)。需包含分组列。")
        st.stop()
        
    raw_df = pd.read_csv(uploaded_file)
    non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if not non_num: st.stop()
    
    group_col = st.selectbox("分组列", non_num)
    grps = raw_df[group_col].unique()
    if len(grps) < 2: st.stop()
    
    case = st.selectbox("Case (Exp)", grps, index=0)
    ctrl = st.selectbox("Control", grps, index=1)
    
    st.divider()
    st.markdown("### ⚙️ 统计参数")
    p_th = st.number_input("P-value Cutoff", 0.05, format="%.3f")
    fc_th = st.number_input("Log2 FC Cutoff", 1.0)
    
# 数据处理
df_proc, feats = preprocess_data(raw_df, group_col)
df_sub = df_proc[df_proc[group_col].isin([case, ctrl])].copy()
res_stats = run_statistics(df_sub, group_col, case, ctrl, feats)

# 标记显著性
res_stats['Sig'] = 'NS'
res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] > fc_th), 'Sig'] = 'Up'
res_stats.loc[(res_stats['P_Value'] < p_th) & (res_stats['Log2_FC'] < -fc_th), 'Sig'] = 'Down'

# 提取显著特征列表
sig_metabolites = res_stats[res_stats['Sig'] != 'NS']['Metabolite'].tolist()

# ==========================================
# 3. 结果展示 (五大模块)
# ==========================================
st.header(f"📊 分析报告: {case} vs {ctrl}")
tabs = st.tabs(["PCA / PLS-DA", "🌋 火山图", "🔥 聚类热图", "🧬 通路富集", "📑 数据表"])

# --- Tab 1: 多变量分析 (PCA & PLS-DA) ---
with tabs[0]:
    col1, col2 = st.columns(2)
    X = StandardScaler().fit_transform(df_sub[feats])
    
    # PCA
    with col1:
        pca = PCA(n_components=2).fit(X)
        pcs = pca.transform(X)
        var = pca.explained_variance_ratio_
        fig_pca = px.scatter(x=pcs[:,0], y=pcs[:,1], color=df_sub[group_col],
                             width=600, height=500)
        # 手动美化点的大小和边框
        fig_pca.update_traces(marker=dict(size=12, line=dict(width=1, color='black')))
        update_layout_pub(fig_pca, "PCA Score Plot", f"PC1 ({var[0]:.1%})", f"PC2 ({var[1]:.1%})")
        st.plotly_chart(fig_pca, use_container_width=True)

    # PLS-DA
    with col2:
        pls = PLSRegression(n_components=2).fit(X, pd.factorize(df_sub[group_col])[0])
        pls_scores = pls.x_scores_
        fig_pls = px.scatter(x=pls_scores[:,0], y=pls_scores[:,1], color=df_sub[group_col],
                             width=600, height=500)
        fig_pls.update_traces(marker=dict(size=12, line=dict(width=1, color='black')))
        update_layout_pub(fig_pls, "PLS-DA Score Plot", "Component 1", "Component 2")
        st.plotly_chart(fig_pls, use_container_width=True)

# --- Tab 2: 火山图 (MetaboAnalyst Style) ---
with tabs[1]:
    # 颜色映射
    color_map = {
        'Up': COLOR_PALETTE['Up'], 
        'Down': COLOR_PALETTE['Down'], 
        'NS': COLOR_PALETTE['NS']
    }
    
    fig_vol = px.scatter(res_stats, x="Log2_FC", y="-Log10_P", color="Sig",
                         color_discrete_map=color_map,
                         hover_data=["Metabolite", "P_Value"],
                         width=800, height=600)
    
    # 增加阈值线
    fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="black", opacity=0.5)
    fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="black", opacity=0.5)
    fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="black", opacity=0.5)
    
    # 样式调整
    fig_vol.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='black')))
    update_layout_pub(fig_vol, "Volcano Plot", "Log2 Fold Change", "-Log10(P-value)")
    
    st.plotly_chart(fig_vol, use_container_width=True)
    st.caption("提示：鼠标悬停右上角相机图标可下载 SVG/PNG 矢量图用于发表。")

# --- Tab 3: 聚类热图 (Seaborn Implementation) ---
with tabs[2]:
    st.subheader("Top 25 显著差异代谢物热图")
    
    if len(sig_metabolites) < 2:
        st.warning("显著差异代谢物太少，无法绘制热图。请尝试放宽 P 值或 FC 阈值。")
    else:
        # 1. 准备数据：取前25个最显著的（按P值排序）
        top_n = 25
        top_feats = res_stats.sort_values('P_Value').head(top_n)['Metabolite'].tolist()
        
        hm_data = df_sub.set_index(group_col)[top_feats]
        
        # 为了画图好看，我们在行（样本）上加颜色条来区分组别
        # 创建一个颜色映射字典
        lut = dict(zip(df_sub[group_col].unique(), "rbg"))
        row_colors = df_sub[group_col].map(lut)
        
        # 2. 绘制 Seaborn Clustermap
        # z_score=1 表示按列（代谢物）进行标准化，这是热图的标准做法
        try:
            g = sns.clustermap(hm_data.astype(float), 
                               z_score=1, 
                               cmap="vlag",  # 红-白-蓝 经典学术配色 (vlag or RdBu_r)
                               center=0, 
                               row_colors=row_colors,
                               figsize=(10, 8),
                               dendrogram_ratio=(.1, .2),
                               cbar_pos=(.02, .32, .03, .2))
            
            # 调整字体
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=10)
            plt.setp(g.ax_heatmap.get_yticklabels(), visible=False) # 隐藏样本名，防止太乱
            
            st.pyplot(g.fig) # 显示 Matplotlib 图
            
        except Exception as e:
            st.error(f"绘图出错 (通常是因为数据量太小): {e}")

# --- Tab 4: 通路富集分析 (Pathway Analysis) ---
with tabs[3]:
    st.subheader("🧬 代谢通路富集 (演示版)")
    
    # 运行通路分析
    path_res = run_pathway_analysis(sig_metabolites, feats)
    
    if path_res.empty:
        st.warning(f"未找到显著富集的通路。这可能是因为演示数据库较小，或者您的代谢物命名与数据库不匹配。\n\n**演示支持的代谢物名**: Glucose, Pyruvate, Lactate, Citrate, Alanine 等。")
    else:
        # 绘制气泡图 (Bubble Plot)
        # X: Impact, Y: -Log10(P), Size: Hits, Color: P-value
        fig_path = px.scatter(path_res, x="Impact", y="-Log10_P",
                              size="Hits", color="P_Value",
                              hover_name="Pathway",
                              size_max=40,
                              color_continuous_scale="Reds_r", # P值越小越红
                              width=800, height=500)
        
        update_layout_pub(fig_path, "Pathway Enrichment Analysis", "Pathway Impact", "-Log10(P-value)")
        
        # 增加文本标签
        fig_path.update_traces(textposition='top center')
        
        st.plotly_chart(fig_path, use_container_width=True)
        
        st.dataframe(path_res)
        st.info("⚠️ 注意：此模块使用内置的小型演示数据库。进行正式发表分析时，请务必使用完整的 KEGG 或 SMPDB 数据库。")

# --- Tab 5: 数据下载 ---
with tabs[4]:
    st.subheader("📥 导出分析结果")
    csv = res_stats.to_csv(index=False).encode('utf-8')
    st.download_button("下载统计结果 (CSV)", csv, "results.csv", "text/csv")

# 文件名: data_preprocessing.py
import pandas as pd
import numpy as np

def data_cleaning_pipeline(df, group_col, 
                           missing_thresh=0.5, 
                           impute_method='min', 
                           norm_method='None', 
                           log_transform=True,
                           scale_method='None'):
    """
    专业的代谢组学数据预处理管道 (Metabolomics Preprocessing Pipeline)
    
    参数:
    ----------
    df : pd.DataFrame
        原始数据，行为样本，列为特征。必须包含分组列。
    group_col : str
        分组列的名称。
    missing_thresh : float (0.0 - 1.0)
        缺失值过滤阈值。如果某列缺失值比例超过此值，将被移除。
        (例如 0.5 表示移除缺失超过 50% 的特征)。
    impute_method : str
        缺失值填充方法: 'min' (1/2 最小值), 'mean', 'median', 'zero'。
    norm_method : str
        样本归一化 (消除稀释效应): 'None', 'Sum' (总和), 'Median' (中位数)。
    log_transform : bool
        是否执行 Log2 转换 (使数据服从正态分布)。
    scale_method : str
        特征标准化: 'None', 'Auto' (Z-score), 'Pareto' (帕累托)。
    
    返回:
    ----------
    processed_df : pd.DataFrame
        处理完成的数据表 (包含元数据)。
    numeric_cols : list
        最终保留的代谢物特征列名列表。
    """
    
    # --- 1. 数据分离 ---
    # 自动识别数值列（代谢物）和非数值列（元数据）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)
    
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    print(f"[Info] 初始特征数: {len(numeric_cols)}")

    # --- 2. 缺失值过滤 (Filter by Missing %) ---
    # 计算每列的缺失比例
    missing_ratio = data_df.isnull().mean()
    # 保留缺失比例小于阈值的列
    cols_to_keep = missing_ratio[missing_ratio <= missing_thresh].index
    data_df = data_df[cols_to_keep]
    
    removed_count = len(numeric_cols) - len(cols_to_keep)
    if removed_count > 0:
        print(f"[Filter] 已移除 {removed_count} 个缺失值超过 {missing_thresh:.0%} 的特征。")
    
    # --- 3. 缺失值填充 (Imputation) ---
    if data_df.isnull().sum().sum() > 0:
        if impute_method == 'min':
            # 使用每列最小值的 1/2 填充 (模拟 LOD)
            min_vals = data_df.min() * 0.5
            data_df = data_df.fillna(min_vals)
        elif impute_method == 'mean':
            data_df = data_df.fillna(data_df.mean())
        elif impute_method == 'median':
            data_df = data_df.fillna(data_df.median())
        elif impute_method == 'zero':
            data_df = data_df.fillna(0)
        
        # 防止整列全空导致的 NaN
        data_df = data_df.fillna(0)

    # --- 4. 样本归一化 (Sample Normalization) ---
    # 目的：消除样本间浓度的差异（如尿液稀释度）
    if norm_method == 'Sum':
        # 除以该样本的总和
        row_sums = data_df.sum(axis=1)
        # 避免除以0
        mean_sum = row_sums.mean()
        data_df = data_df.div(row_sums, axis=0) * mean_sum
    elif norm_method == 'Median':
        # 除以该样本的中位数
        row_medians = data_df.median(axis=1)
        mean_median = row_medians.mean()
        data_df = data_df.div(row_medians, axis=0) * mean_median

    # --- 5. 数据转换 (Transformation) ---
    # 目的：使数据分布接近正态分布，减少异方差性
    if log_transform:
        # Log2(x + 1)
        # 检查负值：如果经过标准化可能出现负数，则不能直接Log
        if (data_df < 0).any().any():
            print("[Warning] 数据包含负数，跳过 Log 转换。")
        else:
            data_df = np.log2(data_df + 1)

    # --- 6. 数据缩放 (Scaling) ---
    # 目的：让高丰度和低丰度代谢物具有可比性 (主要用于 PCA/PLS-DA)
    # 注意：通常箱线图展示的是 Unscaled (仅Log) 的数据，而多变量分析用 Scaled 数据。
    # 这里我们只做计算，具体是否应用取决于用户需求。
    if scale_method == 'Auto':
        # (x - mean) / std
        data_df = (data_df - data_df.mean()) / data_df.std()
    elif scale_method == 'Pareto':
        # (x - mean) / sqrt(std)
        data_df = (data_df - data_df.mean()) / np.sqrt(data_df.std())

    # --- 7. 清理低方差特征 ---
    # 再次检查方差 (防止处理后出现全0列)
    var_mask = data_df.var() > 1e-9 # 设置一个极小的阈值
    data_df = data_df.loc[:, var_mask]
    
    final_cols = data_df.columns.tolist()
    processed_df = pd.concat([meta_df, data_df], axis=1)
    
    print(f"[Info] 最终特征数: {len(final_cols)}")
    return processed_df, final_cols

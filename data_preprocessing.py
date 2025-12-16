import pandas as pd
import numpy as np

def data_cleaning_pipeline(df, group_col, 
                           impute_na=True, 
                           remove_low_var=True, 
                           log_transform=True):
    """
    标准代谢组学数据清洗流程
    
    参数:
    - df: 原始 DataFrame (包含分组列和代谢物数值列)
    - group_col: 分组列的名称
    - impute_na: 是否填充缺失值 (默认填充为最小值的 1/2)
    - remove_low_var: 是否移除方差极低的代谢物 (噪音)
    - log_transform: 是否执行 Log2 变换
    
    返回:
    - processed_df: 处理后的 DataFrame
    - numeric_cols: 数值型列名列表 (代谢物)
    """
    
    # 1. 分离元数据和数值数据
    # 自动识别数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)
    
    meta_cols = [c for c in df.columns if c not in numeric_cols]
    
    data_df = df[numeric_cols].copy()
    meta_df = df[meta_cols].copy()
    
    # 2. 缺失值处理 (Imputation)
    if impute_na:
        # 策略：如果缺失值超过 50%，可能需要删除该列（这里暂不做删除，仅填充）
        # 使用该列最小值的 0.5 倍填充 (模拟检测限 LOD)
        min_vals = data_df.min() * 0.5
        data_df = data_df.fillna(min_vals)
        # 如果整列都是 NaN，填充为 0
        data_df = data_df.fillna(0)

    # 3. 移除低方差特征 (Filtering)
    if remove_low_var:
        # 移除方差为 0 的列 (所有样本数值都一样，没有统计意义)
        var_mask = data_df.var() > 0
        data_df = data_df.loc[:, var_mask]
        numeric_cols = data_df.columns.tolist() # 更新列列表
    
    # 4. 数据转化 (Transformation)
    if log_transform:
        # Log2(x + 1) 处理，避免 log(0)
        # 检查是否包含负数，如果有负数通常不直接做 Log
        if (data_df < 0).any().any():
            # 如果有负数，通常假设数据已经被处理过，跳过 Log
            pass 
        else:
            data_df = np.log2(data_df + 1)

    # 5. 合并返回
    processed_df = pd.concat([meta_df, data_df], axis=1)
    
    return processed_df, numeric_cols

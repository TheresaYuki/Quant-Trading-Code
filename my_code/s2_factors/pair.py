import pandas as pd
import numpy as np
import statsmodels.api as sm

from . import MA

def cal_zscore(series: pd.Series, window: int = 20, shift_period: int = 1) -> pd.Series:
    """
    计算给定序列的滚动 Z-Score (标准分)
    根据传入序列的不同，可计算单/多支股票
    
    参数:
        series (pd.Series): 目标数据序列（比如收盘价、或者两只股票的价差）
        window (int): 滚动窗口大小(默认20天)
        shift_period (int): 滞后周期,防止未来函数(默认1天)
        
    返回:
        pd.Series: Z-Score 序列
    """
    if not isinstance(series, pd.Series):
        raise TypeError("输入的数据必须是 Pandas Series")

    # 1. 计算滚动均值 (Mean)
    rolling_mean = MA.cal_sma(series, window)
    
    # 2. 计算滚动标准差 (Standard Deviation)
    rolling_std = series.rolling(window = window).std()
    
    # 滞后处理,默认数值为1
    if shift_period > 0:
        rolling_mean = rolling_mean.shift(shift_period)
        rolling_std = rolling_std.shift(shift_period)
        
    # 把标准差为0的地方替换成 NaN，避免程序崩溃
    rolling_std = rolling_std.replace(0, np.nan)
    
    # 5. 计算 Z-Score
    z_score = (series - rolling_mean) / rolling_std
    
    return z_score


def cal_pearson(series1: pd.Series, series2: pd.Series) -> float:
    """
    计算两个价格序列（如两只股票的收盘价）的皮尔逊相关系数。
    
    :param series1: 第一只股票的收盘价序列 (pd.Series)
    :param series2: 第二只股票的收盘价序列 (pd.Series)
    :return: 皮尔逊相关系数 (float)
    """
    # 1. 时间对齐：将两个 Series 合并为一个 DataFrame
    # 使用 pd.concat 可以自动按 index（日期）对齐，缺失的日期会自动补 NaN
    df = pd.concat([series1, series2], axis = 1)
    
    # 2. 剔除缺失值：如果某天其中一只股票停牌（为 NaN），则这一天不能参与计算
    df = df.dropna()
    
    # 3. 转换为日收益率
    returns = df.pct_change().dropna()
    
    # 4. 计算皮尔逊相关系数
    corr = returns.iloc[:, 0].corr(returns.iloc[:, 1], method = 'pearson')
    
    return corr


def cal_ols(price_a: pd.Series, price_b: pd.Series):
    """
    使用 OLS 回归计算股票 A 和 B 的对冲比率，并返回价差序列
    """
    # 1. 数据对齐与清洗（剔除缺失值）
    df = pd.DataFrame({'A': price_a, 'B': price_b}).dropna()
    
    # 2. 给自变量 B 添加常数项（截距项 alpha）
    # 如果不加 add_constant，回归线会被强制过原点，这在金融中是不正确的
    X = sm.add_constant(df['B'])
    Y = df['A']
    
    # 3. 执行 OLS 回归
    model = sm.OLS(Y, X).fit()
    
    # 4. 提取回归结果
    alpha = model.params['const']
    beta = model.params['B']
    
    # 5. 计算价差（残差 = 实际价格A - 预测价格A）
    # 预测价格A = alpha + beta * 价格B
    spread = df['A'] - (alpha + beta * df['B'])
    
    # 可选：打印详细的统计报告（理解模型质量）
    # print(model.summary()) 
    
    return {
        'alpha': alpha,
        'beta': beta,
        'spread': spread
    }

# --- 使用示例 ---
# 假设 df_a 和 df_b 是你用 get_data 拉下来的两只高相关 ETF（如 510300 和 510050）
# result = calculate_ols_spread(df_a['close'], df_b['close'])
# spread_series = result['spread']
# hedge_ratio = result['beta']
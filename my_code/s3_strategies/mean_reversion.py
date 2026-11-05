import pandas as pd
import numpy as np

def rsi_signal(df: pd.Series) -> pd.Series:
    """
    根据RSI判断交易信号
    输入:RSI计算结果的列表
    返回一个由(0/1)构成的信号列表
    """
    signals = pd.Series(0, index = df.index)

    signals[df < 30] = 1
    signals[df > 70] = 0

    return signals.ffill().fillna(0)


def boll_bands_signal(df: pd.DataFrame, BB_data: pd.DataFrame, sell_target: str = "BB_Middle") -> pd.Series:
    """
    根据布林带判断交易信号
    输入原数据、处理后的布林带数据、卖出信号(BB_Middle/BB_Upper)
    返回一个由(0/1)构成的信号列表
    """
    signals = pd.Series(0, index = df.index)

    buy_signal = df["close"] < BB_data["BB_Lower"]
    sell_signal = df["close"] > BB_data[sell_target]

    signals[buy_signal] = 1
    signals[sell_signal] = 0

    return signals.ffill().fillna(0)


def grid_signal(df: pd.DataFrame, gap: float, baseline: float, price_col: str = "close") -> pd.Series:
    """
    根据网格交易判断交易信号
    输入包含原数据的表格、网格基线、网格间隔
    输出一个由(-1, 0, 1)构成的交易信号列表(-1卖出, 0不动, 1买入)
    """

    if price_col not in df.columns:
        raise KeyError(f"不存在列：{price_col}")

    # 1. 计算每一刻的价格偏离基准价的幅度
    # 比如价格是 9.5，基准是 10，gap=0.1，那么偏离就是 (9.5-10)/(10*0.1) = -0.5
    deviation = (df[price_col] - baseline) / (baseline * gap)
    
    # 2. 计算当前价格处于哪一个网格层级
    # 使用 np.floor 向下取整。跌得越多，层级越小（负数）；涨得越多，层级越大（正数）
    # 将 np.floor 的结果强制包装为 Pandas Series，并保留原索引
    levels = pd.Series(np.floor(deviation), index = df.index)
    
    # 3. 算出今天和昨天的层级差
    # level_diff > 0：说明涨破网格了；level_diff < 0：说明跌破网格了
    level_diff = levels.diff()
    
    # 4. 生成信号：默认全是 0（不动）
    signals = pd.Series(0, index = df.index)
    
    # 如果层级下降（跌破网格），触发买入（1）
    signals[level_diff < 0] = 1
    
    # 如果层级上升（涨破网格），触发卖出（-1）
    signals[level_diff > 0] = -1
    
    # 第一天的信号处理：因为没有前一天，diff 是 NaN，强制设为 0
    signals.iloc[0] = 0
    
    return signals


# def pair_signal(df1: pd.DataFrame, df2: pd.DataFrame, z_score: float) -> pd.DataFrame:
#     """
#     根据配对交易判断交易信号
#     输入2支股票的原数据表格
#     输出1个由(-1, 0, 1)构成的二维交易信号表格,各自对应2支股票(-1做空, 0空仓, 1做多)
#     """
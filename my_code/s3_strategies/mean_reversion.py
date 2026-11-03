import pandas as pd

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
    输入:原数据、处理后的布林带数据、卖出信号(BB_Middle/BB_Upper)
    返回一个由(0/1)构成的信号列表
    """
    signals = pd.Series(0, index = df.index)

    buy_signal = df["close"] < BB_data["BB_Lower"]
    sell_signal = df["close"] > BB_data[sell_target]

    signals[buy_signal] = 1
    signals[sell_signal] = 0

    return signals.ffill().fillna(0)


def grid_signal(df: pd.DataFrame, gap: float, baseline: float) -> pd.Series:
    """
    根据网格交易判断交易信号
    输入:包含原数据的表格、网格基线、网格间隔
    输出一个由(-1, 0, 1)构成的交易信号列表(-1卖出, 0不动, 1买入)
    """
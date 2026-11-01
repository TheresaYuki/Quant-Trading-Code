import pandas as pd
import numpy as np
from typing import Union

# 计算各种不同类型的MA的函数

def cal_sma(df: Union[pd.DataFrame, pd.Series], window: int, column: str = "close") -> pd.Series:
    """
    计算SMA（求移动窗口的算术平均）
    返回一个一维序列
    """
    if isinstance(df, pd.DataFrame):
        if column not in df.columns:
            raise KeyError(f"不存在列：{column}")
        series = df[column]

    elif isinstance(df, pd.Series):
        series = df

    return series.rolling(window = window).mean()


def cal_ema(df: Union[pd.DataFrame, pd.Series], span: int, column: str = "close") -> pd.Series:
    """
    计算EMA（求移动窗口的指数平均）
    返回一个一维序列
    """

    if isinstance(df, pd.DataFrame):
        if column not in df.columns:
            raise KeyError(f"不存在列：{column}")
        series = df[column]

    elif isinstance(df, pd.Series):
        series = df

    return series.ewm(span = span, adjust = False).mean()


# 这段代码似乎可以用numpy卷积替代apply从而加速运行
def cal_wma(df: Union[pd.DataFrame, pd.Series], window: int, column: str = "close") -> pd.Series:
    """
    计算WMA（求移动窗口的加权平均）
    返回一个一维序列
    """

    if isinstance(df, pd.DataFrame):
        if column not in df.columns:
            raise KeyError(f"不存在列：{column}")
        series = df[column]

    elif isinstance(df, pd.Series):
        series = df

    weights = np.arange(1, window + 1)

    WMA = series.rolling(window = window).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw = True
    )

    return WMA


def cal_rma(df: Union[pd.DataFrame, pd.Series], window: int, column: str = "close") -> pd.Series:
    """
    计算RMA / SMMA（求移动窗口的平滑平均）
    返回一个一维序列
    """
    if isinstance(df, pd.DataFrame):
        if column not in df.columns:
            raise KeyError(f"不存在列：{column}")
        series = df[column]

    elif isinstance(df, pd.Series):
        series = df

    return series.ewm(alpha=1.0 / window, adjust = False).mean()
import pandas as pd
import numpy as np

from . import MA


def cal_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    计算 ATR (平均真实波幅)
    使用 RMA (Wilder's Smoothing) 进行平滑
    """
    if not {"high", "low", "close"}.issubset(df.columns):
        raise KeyError("数据中必须包含 high, low, close 列")

    # 1. 计算 TR (真实波幅)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)

    # 2. 使用 RMA 平滑计算 ATR 
    # 当前只有一列，所以不用填column
    atr = MA.cal_rma(tr, window)
    
    return atr


def cal_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, column: str = "close") -> pd.DataFrame:
    """
    计算布林带 (上轨、中轨、下轨)
    默认按close计算，但也可通过更改column计算其他的
    返回一个包含三列数据的 DataFrame
    """
    if column not in df.columns:
        raise KeyError(f"不存在列：{column}")

    # 1. 计算中轨 (SMA)
    middle_band = MA.cal_sma(df, window, column)
    
    # 2. 计算滚动标准差
    std_dev = df[column].rolling(window = window).std(ddof = 0)
    
    # 3. 计算上下轨
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)

    # 返回拼装好的表格
    return pd.DataFrame({
        "BB_Middle": middle_band,
        "BB_Upper": upper_band,
        "BB_Lower": lower_band
    }, index = df.index)


def cal_hv(df: pd.DataFrame, window: int = 20, column: str = "close") -> pd.Series:
    """
    计算历史波动率 (年化)
    """
    if column not in df.columns:
        raise KeyError(f"不存在列：{column}")

    # 1. 计算日收益率
    daily_returns = df[column].pct_change()
    
    # 2. 计算滚动标准差并年化 (乘以根号252)
    hv = daily_returns.rolling(window = window).std() * np.sqrt(252)
    
    return hv
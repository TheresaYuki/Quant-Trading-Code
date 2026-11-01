import pandas as pd
import numpy as np

from . import MA


# 该代码最后返回的数值设置可能存在某些问题
def cal_rsi(df: pd.DataFrame, window: int = 14, column: str = "close") -> pd.Series:
    """计算RSI指标"""

    delta = df[column].diff()

    gain = delta.clip(lower = 0)
    loss = - delta.clip(upper = 0)

    avg_gain = MA.cal_rma(gain, window)
    avg_loss = MA.cal_rma(loss, window)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 特别注意：这个50可能在某些行情会导致出现极端错误
    return rsi.fillna(50)


def cal_kdj(df: pd.DataFrame, window: int = 9, m1: int = 3, m2: int = 3) -> pd.DataFrame:
    """
    计算 KDJ 指标，返回包含 K, D, J 三列的 DataFrame
    默认参数 N=9, M1=3, M2=3 (通达信标准参数)
    """
    low_n = df["low"].rolling(window = window).min()
    high_n = df["high"].rolling(window = window).max()
    
    # 计算 RSV
    rsv = (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    
    # 计算 K 和 D（A股通常用 RMA 进行平滑）
    # 注意此处
    k = MA.cal_rma(rsv, m1)
    d = MA.cal_rma(k, m2)
    j = 3 * k - 2 * d
    
    return pd.DataFrame({"K": k, "D": d, "J": j}, index = df.index)


def cal_cci(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算 CCI 指标"""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    
    ma_tp = MA.cal_sma(tp, window)

    # 计算平均绝对偏差 (Mean Deviation)
    md = tp.rolling(window = window).apply(lambda x: np.abs(x - x.mean()).mean(), raw = True)
    
    cci = (tp - ma_tp) / (0.015 * md)
    return cci
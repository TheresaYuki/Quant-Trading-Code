import pandas as pd
import numpy as np
from typing import Union

from . import MA

def cal_vma(df: Union[pd.DataFrame, pd.Series], window: int, column: str = "vol") -> pd.Series:
    """计算成交量移动平均 (VMA)"""

    if isinstance(df, pd.DataFrame):
        if column not in df.columns:
            raise KeyError(f"不存在列：{column}")
        series = df[column]

    elif isinstance(df, pd.Series):
        series = df
    
    return MA.cal_sma(series, window)


def cal_vol_ratio(df: Union[pd.DataFrame, pd.Series], window: int, column: str = "vol") -> pd.Series:
    """计算量比"""

    if isinstance(df, pd.DataFrame):
        if column not in df.columns:
            raise KeyError(f"不存在列：{column}")
        series = df[column]

    elif isinstance(df, pd.Series):
        series = df

    return series / cal_vma(series, window)


def cal_turnover(df: pd.DataFrame) -> pd.Series:
    """计算换手率"""
    if "float_shares" not in df.columns:
        raise KeyError("数据缺少 float_shares 列")
    # 如果 float_shares 为 0，用 np.nan 替代防止除以 0 报错
    return df["vol"] / df["float_shares"].replace(0, np.nan)



def cal_obv(df: pd.DataFrame) -> pd.Series:
    """计算能量潮 (OBV)"""
    if "close" not in df.columns or "vol" not in df.columns:
        raise KeyError("数据缺失 close 或 vol 列")
        
    # 用 np.where 判断：涨了为 1，跌了为 -1，平为 0
    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    
    obv = (direction * df["vol"]).cumsum()
    
    return pd.Series(obv, index = df.index)


def cal_mfi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """计算资金流量指标 (MFI)"""
    if not all(col in df.columns for col in ["high", "low", "close", "vol"]):
        raise KeyError("数据缺失 high, low, close 或 vol 列")

    # 1. 典型价格
    tp = (df["high"] + df["low"] + df["close"]) / 3
    
    # 2. 资金流
    mf = tp * df["vol"]
    
    # 3. 区分正负资金流
    # 如果今天 tp > 昨天 tp，就是正，否则是负
    is_positive = tp > tp.shift(1)
    pos_mf = mf.where(is_positive, 0)
    neg_mf = mf.where(~is_positive, 0)
    
    # 4. 过去 N 天的资金流总和
    pos_mf_sum = pos_mf.rolling(window).sum()
    neg_mf_sum = neg_mf.rolling(window).sum()
    
    # 5. 计算 MR（注意：分母为 0 时要替换为 NaN，防止报错）
    mr = pos_mf_sum / neg_mf_sum.replace(0, np.nan)
    
    # 6. 计算 MFI
    mfi = 100 - (100 / (1 + mr))
    
    # 如果 neg_mf_sum 为 0（全涨），MFI 会是 100，保持即可
    return mfi.fillna(100)


def cal_pv_corr(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    计算价量相关性
    返回 -1 到 1 之间的序列
    """

    if "close" not in df.columns or "vol" not in df.columns:
        raise KeyError("数据缺少 close 或 vol 列")
    
    # 为了避免伪相关，通常用“收益率”和“成交量变化率”
    returns = df["close"].pct_change()
    vol_change = df["vol"].pct_change()
    
    # rolling().corr() 自动计算滚动窗口内的相关系数
    pv_corr = returns.rolling(window = window).corr(vol_change)
    
    return pv_corr


def cal_amihud(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    计算 Amihud 非流动性指标
    值越大，说明流动性越差（稍微一买就涨，一卖就跌）
    """

    if "close" not in df.columns or "amount" not in df.columns:
        raise KeyError("数据缺少 close 或 amount 列")
    
    # 1. 绝对收益率
    abs_return = df["close"].pct_change().abs()
    
    # 2. 每日的非流动性 = 绝对收益率 / 成交额
    # 注意：成交额可能是 0（停牌），需要处理防除零
    daily_illiq = abs_return / df["amount"].replace(0, np.nan)
    
    # 3. 滚动求平均
    return MA.cal_sma(daily_illiq, window)
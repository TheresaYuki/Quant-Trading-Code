import pandas as pd
import numpy as np

from . import MA
from . import volatility


def cal_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close"
    ) -> pd.DataFrame:
    """
    计算 MACD 指标
    返回包含 DIF, DEA, MACD_Hist 的二维 DataFrame
    """

    ema_fast = MA.cal_ema(df, span = fast, column = column)
    ema_slow = MA.cal_ema(df, span = slow, column = column)

    dif = ema_fast - ema_slow
    dea = dif.ewm(span = signal, adjust = False).mean()
    macd_hist = (dif - dea) * 2

    return pd.DataFrame({"DIF": dif, "DEA": dea, "MACD_Hist": macd_hist}, index = df.index)


def cal_roc(df: pd.DataFrame, window: int = 20, column: str = "close") -> pd.Series:
    """
    计算变动率 (Rate of Change)
    返回过去 window 天的涨跌幅（小数形式，比如 0.05 代表 5%）
    """
    if column not in df.columns:
        raise KeyError(f"不存在列：{column}")
        
    # pct_change 就是 Pandas 原生的计算涨跌幅函数
    return df[column].pct_change(periods = window)


def cal_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算 ADX / DMI 趋向指标系统。
    
    参数:
        df: 包含 'high', 'low', 'close' 列的 DataFrame
        period: 计算周期，默认 14 天
        
    返回:
        包含 '+DI', '-DI', 'ADX' 列的 DataFrame
    """

    # 1. 防呆设计：检查必要列是否存在
    required_cols = ["high", "low", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"数据中缺少以下列，无法计算 ADX：{missing_cols}")

    # 2. 计算真实波幅 (True Range, TR)
    # 已通过volatility的函数计算，可直接在第4步中获得atr

    # 3. 计算动向移动 (+DM 和 -DM)
    up_move = df["high"] - df["high"].shift(1)
    down_move = df["low"].shift(1) - df["low"]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm = pd.Series(plus_dm, index  = df.index)
    minus_dm = pd.Series(minus_dm, index = df.index)

    # 4. 怀尔德平滑 (RMA)
    plus_dm_smooth = MA.cal_rma(plus_dm, period)
    minus_dm_smooth = MA.cal_rma(minus_dm, period)
    # 直接调用
    atr = volatility.cal_atr(df, period)

    # 5. 计算 +DI 和 -DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)

    # 6. 计算 DX 和 ADX
    # 加上 .replace(0, np.nan) 防止分母为 0 导致报错
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = MA.cal_rma(dx, period)

    # 7. 打包返回 (不污染原 df)
    return pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": adx}, index = df.index)


def cal_bias(df: pd.DataFrame, window: int = 20, column: str = "close") -> pd.Series:
    """
    计算乖离率 (Bias)
    用于评估当前价格偏离均线多远（绝对值越大偏离越远）
    返回一个一维序列
    """

    ma = MA.cal_sma(df, window, column)
    return (df[column] - ma) / ma


def cal_aroon(df: pd.DataFrame, window: int = 25, column: str = "close") -> pd.DataFrame:
    """
    计算这一段时间内的 Aroon 指标 (阿隆指标)
    返回包含 Aroon_Up, Aroon_Down 和 Aroon_Osc 的 DataFrame
    """
    
    if column not in df.columns:
        raise KeyError(f"不存在列：{column}")

    def get_aroon_up(x):
        # np.argmax 返回最大值的索引。如果最后一天是最大值，返回 len-1
        # 距离今天的天数 = (len(x) - 1) - 最大值的索引
        days_since_high = (window - 1) - np.argmax(x)
        return 100 * (window - days_since_high) / window

    def get_aroon_down(x):
        # np.argmin 找最小值的索引
        days_since_low = (window - 1) - np.argmin(x)
        return 100 * (window - days_since_low) / window

    # 计算 Aroon Up 和 Aroon Down
    aroon_up = df[column].rolling(window = window).apply(get_aroon_up, raw = True)
    aroon_down = df[column].rolling(window = window).apply(get_aroon_down, raw = True)
    
    # 计算震荡线 (Oscillator)
    aroon_osc = aroon_up - aroon_down

    return pd.DataFrame({
        "Aroon_Up": aroon_up, 
        "Aroon_Down": aroon_down,
        "Aroon_Osc": aroon_osc
    }, index = df.index)
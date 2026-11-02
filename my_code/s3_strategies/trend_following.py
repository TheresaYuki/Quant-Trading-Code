import pandas as pd
import numpy as np


def dual_moving_average(MA_short: pd.Series, MA_long: pd.Series) -> pd.Series:
    """
    根据短均线和长均线生成双均线仓位信号。
    特别注意:这里传入的MA不一定是SMA的结果
    输入:
        MA_short (pd.Series): 短期均线序列 (如 MA10)
        MA_long (pd.Series): 长期均线序列 (如 MA20)
        
    返回:
        pd.Series: 仓位信号序列。1.0 代表满仓(持有),0.0 代表空仓。
    """

    # 1. 数据校验（防呆）
    if len(MA_short) != len(MA_long):
        raise ValueError("短均线和长均线的长度不一致！")
    if MA_short.index.equals(MA_long.index) is False:
        raise ValueError("短均线和长均线的索引（日期）未对齐！")

    # 2. 寻找金叉和死叉
    # 金叉：今天短 > 长，且昨天短 <= 长
    golden_cross = (MA_short > MA_long) & (MA_short.shift(1) <= MA_long.shift(1))
    
    # 死叉：今天短 < 长，且昨天短 >= 长
    dead_cross = (MA_short < MA_long) & (MA_short.shift(1) >= MA_long.shift(1))

    # 3. 初始化一个全是 NaN 的信号序列（使用短均线的索引）
    signal = pd.Series(np.nan, index = MA_short.index, dtype=float)

    # 4. 将金叉标记为 1.0（买入），死叉标记为 0.0（卖出）
    signal[golden_cross] = 1.0
    signal[dead_cross] = 0.0

    # 5. 向下填充（核心魔法）：把 1.0 或 0.0 的状态一直延续到下一次交叉
    # 初始前几天因为均线是 NaN，交叉不可能发生，所以最后用 0.0 填满
    signal = signal.ffill().fillna(0.0)

    return signal
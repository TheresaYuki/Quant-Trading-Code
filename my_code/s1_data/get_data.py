from easy_tdx import MacClient, Market, Period
import pandas as pd

def get_kline(
        market_str: str,
        code_str: str,
        period_str: str,
        count_int: int
    ) -> pd.DataFrame:
    """
    获取股票/ETF的K线数据
    返回一个表格
    
    返回字段说明:
    - datetime (str): 日期
    - open (float): 开盘价
    - high (float): 最高价
    - low (float): 最低价
    - close (float): 收盘价
    - vol (float): 成交量（股）
    - amount (float): 成交额（元）
    - float_shares (float): 流通份额
    """

    market_map = {
        "SH": Market.SH, #上交所
        "SZ": Market.SZ, #深交所
        "BJ": Market.BJ  #北交所
    }

    period_map = {
        "MIN_1": Period.MIN_1,
        "MIN_5": Period.MIN_5,
        "MIN_15": Period.MIN_15,
        "MIN_30": Period.MIN_30,
        "MIN_60": Period.MIN_60,
        "DAILY": Period.DAILY,
        "WEEKLY": Period.WEEKLY,
        "MONTHLY": Period.MONTHLY,
    }

    market_enum = market_map.get(market_str.upper())
    if market_enum is None:
        raise ValueError(f"不支持的市场：{market_str}")

    period_enum = period_map.get(period_str.upper())
    if period_enum is None:
        raise ValueError(f"不支持的周期：{period_str}")

    with MacClient.from_best_host() as c: 
        df = c.get_stock_kline(
            market = market_enum,
            code = code_str,
            period = period_enum,
            count = count_int
        )

    return df


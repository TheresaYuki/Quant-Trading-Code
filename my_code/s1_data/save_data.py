import pandas as pd
from pathlib import Path

def save_data(df: pd.DataFrame, path: str):
    """
    将数据储存到指定位置
    - 自动处理路径格式
    - 若文件夹不存在，则自动创建
    """

    file_path = Path(path)

    file_path.parent.mkdir(parents = True, exist_ok = True)

    df.to_parquet(path, index = False)

    print(f"parquet类型的数据已成功保存至：{file_path}")
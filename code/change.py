import json
import pandas as pd

# 设置输入输出文件名
input_json = "general_data_fixed.json"       # 原始 JSON 文件
output_csv = "general_data.csv"        # 转换后的 CSV 文件

# 读取 JSON 文件
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data)

# 保存为 CSV
df.to_csv(output_csv, index=False)
print(f"✅ 已成功将 {input_json} 转换为 CSV 文件：{output_csv}")

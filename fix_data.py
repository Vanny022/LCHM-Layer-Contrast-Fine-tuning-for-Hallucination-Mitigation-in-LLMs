# 修复 line-based JSON 文件成合法 JSON 数组
input_file = "general_data.json"
output_file = "general_data_fixed.json"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('[\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if line:
            if i < len(lines) - 1:
                f.write(line + ',\n')
            else:
                f.write(line + '\n')
    f.write(']\n')

print("✅ 已修复 JSON 格式，保存在 general_data_fixed.json")

import yaml
import os

def del_global_cmvn_module(file_path):
    # 读取 YAML 文件为文本
    with open(file_path, "r") as file:
        lines = file.readlines()

    # 删除包含 global_cmvn_module 的段落
    filtered_lines = []
    skip_block = False

    for line in lines:
        # 检测到 global_cmvn_module 时开始跳过
        if "global_cmvn:" in line:
            skip_block = True
        # 如果遇到下一个顶格的字段，结束跳过
        elif skip_block and (not line.startswith(" ") and not line.startswith("-")):
            skip_block = False

        # 如果当前不在跳过状态，保留行
        if not skip_block:
            filtered_lines.append(line)

    # 将结果重新保存成 YAML
    with open(file_path, "w") as file:
        file.writelines(filtered_lines)

    print(f"Updated YAML file saved to {file_path}")


root = "/ssd/zhuang/code/wenet/examples/aishell/paraformer/exp/paraformer"
for file in os.listdir(root):
    if file.endswith(".yaml"):
        del_global_cmvn_module(os.path.join(root, file))

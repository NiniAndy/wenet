
funasr_dict_path = "/ssd/zhuang/code/wenet/examples/cn_correction/dict/tokens.txt"
save_dict_path = "/ssd/zhuang/code/wenet/examples/cn_correction/dict/pny_dict_funasr_style.txt"

with open(funasr_dict_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

with open(save_dict_path, 'w', encoding='utf-8') as file:
    for i in range(len(lines)):
        token = lines[i].strip()
        file.write(f"{token} {i}\n")
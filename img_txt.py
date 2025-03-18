import os
dataset_root='./Dataset'
# 定义四个方向
labels={
    "front": 0,
    "back": 1,
    "left": 2,
    "right": 3,
}
view_dirs = ["front", "back", "left", "right"]
sub_dirs = ["dt", "rt", "at_azimuth", "at_elevation"]

output_txt='./data_list.txt'
with open(output_txt,'w')as f:
    for view_dir in view_dirs:
        file_list = sorted(os.listdir(os.path.join(dataset_root, view_dir, sub_dirs[0])))
        for filename in file_list:
            paths = []
            for sub_dir in sub_dirs:
                file_path=os.path.join(dataset_root,view_dir,sub_dir,filename)
                paths.append(file_path.replace("\\", "/"))  # 统一为正斜杠
            label=labels[view_dir]
            f.write(f"{paths[0]} {paths[1]} {paths[2]} {paths[3]} {label}\n")
print(f"✅ 数据列表已生成: {output_txt}")
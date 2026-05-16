#!/bin/bash

# 指定要打包的文件夹的父文件夹路径
parent_folder=($1)
target_folder=($2)

# 获取所有子文件夹的列表
subfolders=("$parent_folder"/*)

# create the target folder.
if [ ! -d ${target_folder} ]; then
    mkdir ${target_folder}
fi

# 循环遍历子文件夹并打包
for folder in "${subfolders[@]}"; do
  if [ -d "$folder" ]; then
    folder_name=$(basename "$folder")
    tar -czvf "${target_folder}/${folder_name}.tar.gz" "$folder"
    echo "Pack $folder_name"
  fi
done

echo "All pack success!"

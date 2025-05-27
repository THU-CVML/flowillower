#!/bin/bash

# 获取脚本所在绝对路径
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 源路径（脚本所在目录）
SOURCE_DIR="$SCRIPT_DIR"

# 目标路径（相对脚本目录的../../flowillower/）
TARGET_DIR="$SCRIPT_DIR/../../flowillower"

# 自动创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR" || { echo "无法创建目录 $TARGET_DIR"; exit 1; }

# 执行复制（递归复制并显示进度）
echo "正在将代码从 $SOURCE_DIR 复制到 $TARGET_DIR"
rsync -ah --progress "$SOURCE_DIR/" "$TARGET_DIR/" && echo "复制完成" || echo "复制失败"

# 如果系统没有 rsync，可以用 cp 替代：
# cp -rv "$SOURCE_DIR"/* "$TARGET_DIR/" && echo "复制完成" || echo "复制失败"
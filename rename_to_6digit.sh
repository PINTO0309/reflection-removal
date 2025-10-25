#!/bin/bash
set -e

# === 設定 ===
TARGET_DIR="${1:-.}"  # 引数があればそのフォルダ、なければカレントディレクトリ
cd "$TARGET_DIR"

# === 拡張子 .jpg (または .JPG) のファイルを取得してソート ===
files=($(ls -1v *.jpg 2>/dev/null || ls -1v *.JPG 2>/dev/null))

if [ ${#files[@]} -eq 0 ]; then
  echo "No .jpg files found in $TARGET_DIR"
  exit 1
fi

echo "Renaming ${#files[@]} files in $TARGET_DIR ..."

# === 一時的な衝突回避リネーム ===
for f in "${files[@]}"; do
  mv "$f" "_tmp_$f"
done

# === 4桁連番にリネーム ===
count=1
for f in _tmp_*.jpg _tmp_*.JPG; do
  newname=$(printf "2%05d.jpg" "$count")
  mv "$f" "$newname"
  ((count++))
done

echo "Done. Renamed ${count-1} files."

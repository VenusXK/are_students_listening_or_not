# 基于 `yolov5` 的学生学习状态实时分析

## 文件说明

- `yolo(backup)` 内包含备份的 `yolo` 文件
  - `detect20230216.py` 文件为移动应用开发课程课程设计实现的推理脚本，其功能如下
    - 收流 `ffmpeg`
    - 推流 `python` `pipe` 线程
    - 分析 基本的 `yolo` `detect.py` 功能
    - 分析数据写入数据库
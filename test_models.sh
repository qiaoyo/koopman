#!/bin/bash

# 定义窗口大小数组
windows=( 100)

# 定义模型名称数组
models=('DronePose')

# 遍历每个模型
for model in "${models[@]}"; do
    echo "开始测试模型: $model"
    
    # 为每个窗口大小启动一个后台进程
    for window in "${windows[@]}"; do
        echo "启动 $model 的 window=$window 测试进程"
        # 使用setsid在后台运行，并将输出重定向到日志文件
        setsid python test_model_wandb.py -w $window -m $model > "test_logs/${model}_w${window}_test.log" 2>&1 &
    done
    
    # 等待所有后台进程完成
    echo "等待 $model 的所有测试进程完成..."
    wait
    
    echo "$model 的所有测试已完成"
    echo "----------------------------------------"
done

echo "所有模型测试完成！" 
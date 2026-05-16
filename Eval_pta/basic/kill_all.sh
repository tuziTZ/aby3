#!/bin/bash

if [ ! -n "$1" ]; then
    echo "offer keyword: "
    read keyword
    keyword=${keyword}
else
    keyword=$1
fi

if [ ! -n "$2" ]; then
    repeat=1
else
    repeat=$2
fi

# 查找包含关键词的进程的PID  
pids=$(ps aux | grep "$keyword" | awk '{print $2}')  
  
# 杀死所有找到的进程  
for pid in $pids; do  
    for i in {1...$repeat}; do
        kill -9 $pid
    done;
done

targets=(aby31 aby32);
for target_machine in ${targets[@]}; do
    echo "in here"
    ssh $target_machine "sh ./aby3/Eval/basic/kill_all.sh ${keyword}";
done
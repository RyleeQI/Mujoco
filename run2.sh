#-w node02
#-p RTX2080Ti
# -p RTX3090
num=2
#Debug
# srun -J sample_true -N 1 -p RTX2080Ti -w node02 --gres gpu:$num bash shell_run/task_TD3_BC.sh -d -g $num
#Train
srun -J sample_true -N 1 -p RTX2080Ti -w node08 --gres gpu:$num bash shell_run/task_TD3_BC2.sh -g $num

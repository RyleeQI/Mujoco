#-w node02
#-p RTX2080Ti
# -p RTX3090
num=3
#Debug
#srun -J sample_true -N 1 -p RTX2080Ti -w node02 --gres gpu:$num bash shell_run/task_TD3_BC1.sh -d -g $num
#Train
# srun -J Res_action -N 1 -p -w node01 --gres gpu:$num 
bash shell_run/task_TD3_BC1.sh -g $num

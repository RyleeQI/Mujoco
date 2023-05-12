if $debug ; then
  export WANDB_MODE='offline'
  ENABLE_wandb=0
  echo "Debugging brach: $debug ; ENABLE_wandb $ENABLE_wandb , 1 means wandb is enabled"
else
  #TODO 待实现：git 上传
  ENABLE_wandb=1
  echo "Not_debugging brach: $debug ; ENABLE_wandb $ENABLE_wandb"
fi
#TODO 自动本地记录
[ -z "$entry_file" ] && entry_file=BC_main.py
# /cluster/home1/yqs/anaconda3/envs/d4rl/bin/python $entry_file $exp_args "${@}" --en_wandb $ENABLE_wandb
python $entry_file $exp_args "${@}" --en_wandb $ENABLE_wandb
sleep 1
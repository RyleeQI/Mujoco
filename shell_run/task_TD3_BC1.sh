
#单张卡上最多跑几个实验
parallel_per_gpu=5

. shell_run/parse.sh
echo "Debug value check: $debug ,true means debugging"
echo "GPU num check: $gpus ,applied $gpus gpus"

#申请到的GPU数量
begin=0
end=$((begin + gpus - 1))
echo "end_num:$end"
#重复实验的次数
para=1
envs=(
	"halfcheetah-random-v2"
	"hopper-random-v2"
	"walker2d-random-v2"
	"halfcheetah-medium-v2"
	"hopper-medium-v2"
	"walker2d-medium-v2"
	"halfcheetah-expert-v2"
	"hopper-expert-v2"
	"walker2d-expert-v2"
	# "halfcheetah-medium-expert-v2"
	# "hopper-medium-expert-v2"
	# "walker2d-medium-expert-v2"
	"halfcheetah-medium-replay-v2"
	"hopper-medium-replay-v2"
	"walker2d-medium-replay-v2"
	)

envs1=(
	"halfcheetah-random-v2"
	"hopper-random-v2"
	"walker2d-random-v2"
	"halfcheetah-medium-v2"
	"hopper-medium-v2"
	"walker2d-medium-v2"
	"halfcheetah-expert-v2"
	"hopper-expert-v2"
	"walker2d-expert-v2"
	"halfcheetah-medium-expert-v2"
	"hopper-medium-expert-v2"
	"walker2d-medium-expert-v2"
	"halfcheetah-medium-replay-v2"
	"hopper-medium-replay-v2"
	"walker2d-medium-replay-v2"
	)
seedset=(0 1 2 3 4)

# 公共超参数 意思是同一个group共有的参数？
export exp_args="--tag baseline_BC_0726 "
. shell_run/utils.sh
for env in ${envs1[*]}
do 
	for seed in ${seedset[*]}
	do
		run --env $env --seed $seed
	done
done
wait


# Bark TODO
# replace the token with yours
if [ $? -eq 0 ]; then
   curl "https://api.day.app/yFSpfNMfaTmCQkELJTnHjj/Finished/($exp_args)"
else
   curl "https://api.day.app/yFSpfNMfaTmCQkELJTnHjj/Failed/($exp_args)"
fi

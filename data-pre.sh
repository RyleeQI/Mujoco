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
	"halfcheetah-medium-expert-v2"
	"hopper-medium-expert-v2"
	"walker2d-medium-expert-v2"
	"halfcheetah-medium-replay-v2"
	"hopper-medium-replay-v2"
	"walker2d-medium-replay-v2"
	)

for env in ${envs[*]}
do 
    python data-pre.py --env $env &
done
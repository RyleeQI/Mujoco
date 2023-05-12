[ -z "$begin" ] && begin=0
[ -z "$end" ] && end=7
[ -z "$parallel_per_gpu" ] && parallel_per_gpu=4
[ -z "$para" ] && para=1

max_gpus=$((end - begin + 1))

# [ -n "$1" ] && export debug=true

max_run=$((max_gpus * parallel_per_gpu))
echo max_run is $max_run from "[$begin, $end]"
counter=0
run() {
  for i in $(seq 1 $para); do
    echo counter $counter
    export CUDA_VISIBLE_DEVICES=$((counter % max_gpus + begin))
    if $debug; then
      echo "debug brach: serial $debug"
      bash shell_run/entry.sh "${@}"
    else
      echo "not_debug brach: parallel $debug"
      bash shell_run/entry.sh "${@}" &
    fi
    sleep 1
    counter=$((counter + 1))
    if [[ $((counter % max_run)) -eq 0 ]]; then wait; fi
  done
}



POSITIONAL=()
export debug=false
# $#是变量的个数
# -gt是大于
# 将所有的参数过一遍，输入参数是--numworkers 2 --batchsize 1024这样的形式
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -d|--debug)
      export debug=true
      shift # past argument
      ;;
    -t|--test)
      export test=true
      shift # past value
      ;;
    -g|--gres)
      export gpus="$2"
      shift # past value
      shift # past value
      ;;
    --lr)
      export lr="$2"
      shift # past value
      shift # past value
      ;;
    -b|--batch_size)
      export batch_size="$2"
      shift # past value
      shift # past value
      ;;
    -j|--num_workers)
      export num_workers="$2"
      shift # past value
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

echo POSITIONAL ARGS:
for i in ${POSITIONAL[@]}; do
  echo $i
done

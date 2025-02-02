#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -W ignore -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/extract_pseudolabel.py $CONFIG --launcher pytorch ${@:4}

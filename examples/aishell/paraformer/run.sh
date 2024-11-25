#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

stage=0
stop_stage=1

# You should change the following two parameters for multiple machine training,
# see https://pytorch.org/docs/stable/elastic/run.html
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2024

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
#data_type=shard
data_type=raw

#train_config=conf/train_paraformer.yaml
#dir=exp/paraformer
train_config=conf/train_paraformer_V2.yaml
dir=exp/paraformerV2
tensorboard_dir=tensorboard
num_workers=8
prefetch=500

# use average_checkpoint will get better result
average_checkpoint=true
average_num=10
decode_checkpoint=$dir/avg_${average_num}.pt
decode_modes="ctc_greedy_search ctc_prefix_beam_search paraformer_greedy_search"

train_engine=torch_ddp

# model+optimizer or model_only, model+optimizer is more time-efficient but
# consumes more space, while model_only is the opposite
deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model_only"

. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  # NOTE(xcsong): deepspeed fails with gloo, see
  #   https://github.com/microsoft/DeepSpeed/issues/2818
  dist_backend="nccl"

  current_time=$(date "+%Y-%m-%d_%H-%M")
  log_file="${dir}/train.log.txt.${current_time}"
  echo "log_file: ${log_file}"

  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type  $data_type \
      --train_data /ssd/zhuang/code/wenet/examples/aishell/s0/data/train/data.list \
      --cv_data /ssd/zhuang/code/wenet/examples/aishell/s0/data/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states} &> ${log_file}
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    # 如果不存在平均模型，则先进行平均
    if [ ! -f $decode_checkpoint ]; then
      echo "do model average and final checkpoint is $decode_checkpoint"
      python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $dir  \
        --num ${average_num} \
        --val_best
    fi
  else
    echo "do not do model average and final checkpoint is $checkpoint"
    decode_checkpoint=$dir/$checkpoint
  fi
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  decoding_chunk_size=
  ctc_weight=0.3
  reverse_weight=0.0

  result_dir="${decode_checkpoint}-ctc${ctc_weight}-re${reverse_weight}-inference"
  mkdir -p $result_dir

  python wenet/bin/recognize.py --gpu 0 \
    --modes $decode_modes \
    --config $dir/train.yaml \
    --data_type $data_type \
    --test_data /ssd/zhuang/code/wenet/examples/aishell/s0/data/test/data.list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 32 \
    --blank_penalty 0.0 \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $result_dir \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
  for mode in ${decode_modes}; do
    python tools/compute-wer.py --char=1 --v=1 \
      /ssd/zhuang/code/wenet/examples/aishell/s0/data/test/text $result_dir/$mode/text > $result_dir/$mode/wer
  done
fi


#if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#  # Export the best model you want
#  python wenet/bin/export_jit.py \
#    --config $dir/train.yaml \
#    --checkpoint $dir/avg_${average_num}.pt \
#    --output_file $dir/final.zip \
#    --output_quant_file $dir/final_quant.zip
#fi

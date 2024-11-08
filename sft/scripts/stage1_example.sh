set -ex

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

BATCH_SIZE=4096
MICRO_BATCH_SIZE=8
GRAD_ACCU=$(($BATCH_SIZE / $WORLD_SIZE / $MICRO_BATCH_SIZE / 8))

DATA_PATH=" |your data path| "
OUTPUT_PATH=" |output path| "
MODEL_PATH=" |model path| "
LOG_PATH=" |log path| "

mkdir -p $OUTPUT_PATH
mkdir -p $LOG_PATH

cd /tools
torchrun ${DISTRIBUTED_ARGS} finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 1216 \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRAD_ACCU \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --logging_dir $LOG_PATH \
    --deepspeed configs/stage3.json \
    --bf16 True
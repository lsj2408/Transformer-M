ulimit -c unlimited

# example for HOMO (task_idx=2) & LUMO (task_idx=3)

[ -z "${lr}" ] && lr=7e-5
[ -z "${end_lr}" ] && end_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=60000
[ -z "${total_steps}" ] && total_steps=600000
[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=32
[ -z "${batch_size}" ] && batch_size=128
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=5
[ -z "${data_path}" ] && data_path='./'
[ -z "${save_path}" ] && save_path='./logs'
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln="false"
[ -z "${droppath_prob}" ] && droppath_prob=0.0

[ -z "${loss_type}" ] && loss_type="L2"
[ -z "${std_type}" ] && std_type="std_logits"
[ -z "${readout_type}" ] && readout_type="cls"

[ -z "${save_interval}" ] && save_interval=10
[ -z "${dataset_name}" ] && dataset_name="qm9H"

[ -z "${task_idx}" ] && task_idx=2

[ -z "${add_3d}" ] && add_3d="true"
[ -z "${no_2d}" ] && no_2d="true"
[ -z "${no_save}" ] && no_save="false"
[ -z "${no_pretrain}" ] && no_pretrain="false"
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128

[ -z "${MASTER_PORT}" ] && MASTER_PORT=10086
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1

[ -z "${save_prefix}" ] && save_prefix='exp'
[ -z "${ckpt_path}" ] && ckpt_path='../ckpts/ckpt.pt' # set this dir

echo -e "\n\n"
echo "==================================MP==========================================="
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "OMPI_COMM_WORLD_RANK: ${OMPI_COMM_WORLD_RANK}"
echo "OMPI_COMM_WORLD_SIZE: ${OMPI_COMM_WORLD_SIZE}"


if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  ddp_options=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
	ddp_options=""
  else
    ddp_options="--nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR"
  fi
fi
echo "ddp_options: ${ddp_options}"
echo "==============================================================================="

hyperparams=dataset-$dataset_name-lr-$lr-end_lr-$end_lr-tsteps-$total_steps-wsteps-$warmup_steps-L$layers-D$hidden_size-F$ffn_size-H$num_head-SLN-$sandwich_ln-BS$((batch_size*n_gpu*OMPI_COMM_WORLD_SIZE*update_freq))-CLIP$clip_norm-dp$dropout-attn_dp$attn_dropout-wd$weight_decay-dpp$droppath_prob/SEED$seed-TASK$task_idx-LOSS-$loss_type-STD-$std_type-RF-$readout_type
save_dir=$save_path/$save_prefix-$hyperparams
tsb_dir=$save_dir/tsb
mkdir -p $save_dir

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "seed: ${seed}"
echo "batch_size: $((batch_size*n_gpu*OMPI_COMM_WORLD_SIZE*update_freq))"
echo "n_layers: ${layers}"
echo "lr: ${lr}"
echo "warmup_steps: ${warmup_steps}"
echo "total_steps: ${total_steps}"
echo "clip_norm: ${clip_norm}"
echo "hidden_size: ${hidden_size}"
echo "ffn_size: ${ffn_size}"
echo "sandwich_ln: ${sandwich_ln}"
echo "num_head: ${num_head}"
echo "update_freq: ${update_freq}"
echo "dropout: ${dropout}"
echo "attn_dropout: ${attn_dropout}"
echo "act_dropout: ${act_dropout}"
echo "weight_decay: ${weight_decay}"
echo "droppath_prob: ${droppath_prob}"
echo "save_dir: ${save_dir}"
echo "tsb_dir: ${tsb_dir}"
echo "data_dir: ${data_path}"
echo "ckpt_dir: ${ckpt_path}"
echo "task_idx: ${task_idx}"
echo "loss_type: ${loss_type}"
echo "std_type: ${std_type}"
echo "readout_type: ${readout_type}"
echo "save_interval: ${save_interval}"
echo "dataset_name: ${dataset_name}"
echo "==============================================================================="

# ENV
echo -e "\n\n"
echo "======================================ENV======================================"
echo 'Environment'
ulimit -c unlimited;
echo '\n\nhostname'
hostname
echo '\n\nnvidia-smi'
nvidia-smi
echo '\n\nls -alh'
ls -alh
echo -e '\n\nls ~ -alh'
ls ~ -alh
echo "torch version"
python -c "import torch; print(torch.__version__)"
echo "==============================================================================="

echo -e "\n\n"
echo "==================================ACTION ARGS==========================================="
if ( $sandwich_ln == "true")
then
  action_args="--sandwich-ln "
else
  action_args=""
fi
echo "action_args: ${action_args}"

if ( $add_3d == "true")
then
  add_3d_args="--add-3d"
else
  add_3d_args=""
fi
echo "add_3d_args: ${add_3d_args}"

if ( $no_2d == "true")
then
  no_2d_args="--no-2d"
else
  no_2d_args=""
fi
echo "no_2d_args: ${no_2d_args}"

if ( $no_save == "true" )
then
  no_save_args="--no-epoch-checkpoints"
else
  no_save_args=""
fi
echo "no_save_args: ${no_save_args}"

if ( $no_pretrain == "true" )
then
  load_pretrain_args=""
else
  load_pretrain_args="--remove-head --finetune-from-model ${ckpt_path}"
fi
echo "load_pretrain_args: ${load_pretrain_args}"

echo "========================================================================================"

export NCCL_ASYNC_ERROR_HADNLING=1
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $ddp_options train.py \
	--user-dir $(realpath ./Transformer-M) \
	--data-path $data_path \
	--num-workers 16 --ddp-backend=legacy_ddp \
	--dataset-name $dataset_name --valid-subset valid,test \
	--batch-size $batch_size --data-buffer-size 20 --seed $seed \
	--task graph_prediction_qm9 --criterion graph_prediction_qm9 --arch transformer_m --load-qm9 --remove-head --num-classes 1 \
	--lr $lr --end-learning-rate $end_lr --lr-scheduler polynomial_decay --power 1 \
	--warmup-updates $warmup_steps --total-num-update $total_steps --max-update $total_steps --update-freq $update_freq \
	--encoder-layers $layers --encoder-attention-heads $num_head $add_3d_args $no_2d_args --num-3d-bias-kernel $num_3d_bias_kernel \
	--encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size --droppath-prob $droppath_prob \
	--attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout --weight-decay $weight_decay \
	--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 $action_args --clip-norm $clip_norm $no_save_args \
	--loss-type $loss_type --std-type $std_type --readout-type $readout_type \
	--tensorboard-logdir $tsb_dir --save-dir $save_dir --task-idx $task_idx $load_pretrain_args --save-interval $save_interval | tee $save_dir/train_log.txt

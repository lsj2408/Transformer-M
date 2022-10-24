ulimit -c unlimited

[ -z "${layers}" ] && layers=12
[ -z "${hidden_size}" ] && hidden_size=768
[ -z "${ffn_size}" ] && ffn_size=768
[ -z "${num_head}" ] && num_head=32
[ -z "${batch_size}" ] && batch_size=256
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=5
[ -z "${data_path}" ] && data_path='./datasets/'
[ -z "${save_path}" ] && save_path='./logs/path_to_ckpts/'
[ -z "${dropout}" ] && dropout=0.0
[ -z "${act_dropout}" ] && act_dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${sandwich_ln}" ] && sandwich_ln="false"
[ -z "${droppath_prob}" ] && droppath_prob=0.1
[ -z "${noise_scale}" ] && noise_scale=0.2
[ -z "${mode_prob}" ] && mode_prob="0.2,0.2,0.6"

[ -z "${dataset_name}" ] && dataset_name="PCQM4M-LSC-V2-3D"
[ -z "${add_3d}" ] && add_3d="true"
[ -z "${no_2d}" ] && no_2d="false"
[ -z "${num_3d_bias_kernel}" ] && num_3d_bias_kernel=128

python evaluate.py \
	--user-dir $(realpath ./Transformer-M) \
	--data-path $data_path \
	--num-workers 16 --ddp-backend=legacy_ddp \
	--dataset-name $dataset_name \
	--batch-size $batch_size --data-buffer-size 20 \
	--task graph_prediction --criterion graph_prediction --arch transformer_m_base --num-classes 1 \
	--encoder-layers $layers --encoder-attention-heads $num_head --add-3d --num-3d-bias-kernel $num_3d_bias_kernel \
	--encoder-embed-dim $hidden_size --encoder-ffn-embed-dim $ffn_size --droppath-prob $droppath_prob \
	--attention-dropout $attn_dropout --act-dropout $act_dropout --dropout $dropout \
	--save-dir $save_path --noise-scale $noise_scale --mode-prob $mode_prob --split valid --metric mae

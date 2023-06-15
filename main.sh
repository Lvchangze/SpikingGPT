python -u main.py \
    --seed 42 \
    --dataset_path "/home/ltl/snn_gpt/红楼梦.txt" \
    --ckpt_path "/home/ltl/save_model000.pth" \
    --block_size 64 \
    --time_step 4 \
    --hidden_dim 512 \
    --num_heads 8 \
    --depths 12 \
    --epochs 1000 \
    --batch_size 128 \
    --learning_rate 3e-4 \
    --grad_norm_clip 0.5 \
    --lr_decay 0.1 \
    --context "林黛玉哭着说" \
    --steps 2000 \
    --temperature 1.0 \
    --sample False \
    --top_k 50 \
    > run1000.log

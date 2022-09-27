#!/bin/bash

low_ppc=(1 4 8 16 32 80 160 320)
high_ppc=(4 8 16 32 80 160 320 1600)
for i in "${!low_ppc[@]}"; do
	lp="${low_ppc[i]}"
	hp="${high_ppc[i]}"
	echo "training $lp -> $hp model"
	python3 scripts/train_cnn.py --batch_size 8 --epochs 3000 --lr 2e-3 --alpha 0.75 --im_channels 1 --hidden_channels 2 --num_res_blocks 3 --low_ppc $lp --high_ppc $hp --sim_dir out/ --var jx --im_size 125 133 --save_path models/ --model_name convxcoder/$lp-$hp-3l-2c.pt
done

#!/bin/bash

channels=2
resblocks=3
batchsize=8
epochs=2000

low_ppc=(320 160) # 80 32 16 8 4 1)
high_ppc=(1600 320) # 160 80 32 16 8 4)

for i in "${!low_ppc[@]}"; do
	lp="${low_ppc[i]}"
	hp="${high_ppc[i]}"
	echo "training $lp -> $hp model"
	python3 scripts/train_cnn.py --batch_size $batchsize --epochs $epochs --lr 2e-3 --alpha 0.75 --im_channels 1 --hidden_channels $channels --num_res_blocks $resblocks --low_ppc $lp --high_ppc $hp --sim_dir out/ --var jx --im_size 125 133 --save_path models/ --model_name convxcoder/$lp-$hp-${resblocks}l-${channels}c-tc.pt --tencrop
done

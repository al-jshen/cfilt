#!/bin/bash

channels=8
resblocks=3
batchsize=64
epochs=5000

low_ppc=(4)
high_ppc=(1600)

for i in "${!low_ppc[@]}"; do
	lp="${low_ppc[i]}"
	hp="${high_ppc[i]}"
	echo "training $lp -> $hp model"
	python3 scripts/train_cnn.py --batch_size $batchsize --epochs $epochs --lr 2e-3 --alpha 0.7 --im_channels 1 --hidden_channels $channels --num_res_blocks $resblocks --low_ppc $lp --high_ppc $hp --sim_dir out/ --var jx --im_size 125 133 --save_path models --model_name convxcoder/$lp-$hp-${resblocks}l-${channels}c-new.pt --train --verbose
done

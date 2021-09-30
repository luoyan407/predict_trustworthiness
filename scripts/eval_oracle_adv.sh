#!/bin/bash
DATA='/path-to-data/'
PRETRAINED_MODEL='pretrained/imagenet21k+imagenet2012_ViT-B_16.pth'
python src/eval_oracle_adversarial.py \
	--exp-name eval_oracle_ovit_cvit_ce_adv \
	--n-gpu 1 \
	--classifier 'transformer' \
	--image-size 384 \
	--oracle-type 'transformer' \
	--oracle-feat-dim 768 \
	--oracle-loss 'ce' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 10 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 2 \
	--warmup-steps 100 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL} \
	--oracle-pretrained '/path-to-the-experiment-folder/checkpoints/ep_01.pth'
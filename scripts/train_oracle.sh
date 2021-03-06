#!/bin/bash
DATA='/path-to-data/'
PRETRAINED_MODEL='pretrained/imagenet21k+imagenet2012_ViT-B_16.pth'
#------------<O: ViT, C: ViT>------------
python src/train_oracle.py \
	--exp-name oracle_ovit_cvit_ce \
	--n-gpu 4 \
	--classifier 'transformer' \
	--image-size 384 \
	--oracle-type 'transformer' \
	--oracle-feat-dim 768 \
	--oracle-loss 'ce' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL}
python src/train_oracle.py \
	--exp-name oracle_ovit_cvit_focal \
	--n-gpu 4 \
	--classifier 'transformer' \
	--image-size 384 \
	--oracle-type 'transformer' \
	--oracle-feat-dim 768 \
	--oracle-loss 'focal' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL} \
	--oracle-loss-hyperparam 2.0 
python src/train_oracle.py \
	--exp-name oracle_ovit_cvit_tcp \
	--n-gpu 4 \
	--classifier 'transformer' \
	--image-size 384 \
	--oracle-type 'transformer' \
	--oracle-feat-dim 768 \
	--oracle-loss 'tcp' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL}
python src/train_oracle.py \
	--exp-name oracle_ovit_cvit_steep \
	--n-gpu 4 \
	--classifier 'transformer' \
	--image-size 384 \
	--oracle-type 'transformer' \
	--oracle-feat-dim 768 \
	--oracle-loss 'steep' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL} \
	--oracle-loss-hyperparam 3.0 1.0 0.0 0.0
#------------<O: ViT, C: ResNet>------------
python src/train_oracle.py \
	--exp-name oracle_ovit_crsn_ce \
	--n-gpu 4 \
	--classifier 'resnet' \
	--image-size 224 \
	--oracle-type 'transformer' \
	--oracle-feat-dim 768 \
	--oracle-loss 'ce' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL}
python src/train_oracle.py \
	--exp-name oracle_ovit_crsn_focal \
	--n-gpu 4 \
	--classifier 'resnet' \
	--image-size 224 \
	--oracle-type 'transformer' \
	--oracle-feat-dim 768 \
	--oracle-loss 'focal' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL} \
	--oracle-loss-hyperparam 2.0 
python src/train_oracle.py \
	--exp-name oracle_ovit_crsn_tcp \
	--n-gpu 4 \
	--classifier 'resnet' \
	--image-size 224 \
	--oracle-type 'transformer' \
	--oracle-feat-dim 768 \
	--oracle-loss 'tcp' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL}
python src/train_oracle.py \
	--exp-name oracle_ovit_crsn_steep \
	--n-gpu 4 \
	--classifier 'resnet' \
	--image-size 224 \
	--oracle-type 'transformer' \
	--oracle-feat-dim 768 \
	--oracle-loss 'steep' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL} \
	--oracle-loss-hyperparam 3.0 1.0 0.0 0.0
#------------<O: ResNet, C: ViT>------------
python src/train_oracle.py \
	--exp-name oracle_orsn_cvit_ce \
	--n-gpu 4 \
	--classifier 'transformer' \
	--image-size 384 \
	--oracle-type 'resnet' \
	--oracle-feat-dim 2048 \
	--oracle-loss 'ce' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL}
python src/train_oracle.py \
	--exp-name oracle_orsn_cvit_focal \
	--n-gpu 4 \
	--classifier 'transformer' \
	--image-size 384 \
	--oracle-type 'resnet' \
	--oracle-feat-dim 2048 \
	--oracle-loss 'focal' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL} \
	--oracle-loss-hyperparam 2.0 
python src/train_oracle.py \
	--exp-name oracle_orsn_cvit_tcp \
	--n-gpu 4 \
	--classifier 'transformer' \
	--image-size 384 \
	--oracle-type 'resnet' \
	--oracle-feat-dim 2048 \
	--oracle-loss 'tcp' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL}
python src/train_oracle.py \
	--exp-name oracle_orsn_cvit_steep \
	--n-gpu 4 \
	--classifier 'transformer' \
	--image-size 384 \
	--oracle-type 'resnet' \
	--oracle-feat-dim 2048 \
	--oracle-loss 'steep' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL} \
	--oracle-loss-hyperparam 5.0 2.0 0.0 0.0
#------------<O: ResNet, C: ResNet>------------
python src/train_oracle.py \
	--exp-name oracle_orsn_crsn_ce \
	--n-gpu 4 \
	--classifier 'resnet' \
	--image-size 224 \
	--oracle-type 'resnet' \
	--oracle-feat-dim 2048 \
	--oracle-loss 'ce' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL}
python src/train_oracle.py \
	--exp-name oracle_orsn_crsn_focal \
	--n-gpu 4 \
	--classifier 'resnet' \
	--image-size 224 \
	--oracle-type 'resnet' \
	--oracle-feat-dim 2048 \
	--oracle-loss 'focal' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL} \
	--oracle-loss-hyperparam 2.0 
python src/train_oracle.py \
	--exp-name oracle_orsn_crsn_tcp \
	--n-gpu 4 \
	--classifier 'resnet' \
	--image-size 224 \
	--oracle-type 'resnet' \
	--oracle-feat-dim 2048 \
	--oracle-loss 'tcp' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL}
python src/train_oracle.py \
	--exp-name oracle_orsn_crsn_steep \
	--n-gpu 4 \
	--classifier 'resnet' \
	--image-size 224 \
	--oracle-type 'resnet' \
	--oracle-feat-dim 2048 \
	--oracle-loss 'steep' \
	--model-arch b16 \
	--checkpoint-path ${PRETRAINED_MODEL} \
	--batch-size 40 \
	--tensorboard \
	--data-dir ${DATA} \
	--dataset ImageNet \
	--num-classes 1000 \
	--train-epochs 1 \
	--lr 1e-5 \
	--wd 0 \
	--momentum 0.05 \
	--oracle-model-arch b16 \
	--oracle-checkpoint-path ${PRETRAINED_MODEL} \
	--oracle-loss-hyperparam 5.0 2.0 0.0 0.0
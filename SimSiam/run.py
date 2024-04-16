# -*- coding=utf-8 -*-
import os

os.system('python main.py \
--dataset raf \
--image_size 224 \
--model simsiam \
--proj_layers 2 \
--backbone resnet18 \
--optimizer sgd \
--weight_decay 0.0005 \
--momentum 0.9 \
--warmup_epoch 10 \
--warmup_lr 0 \
--base_lr 0.03 \
--final_lr 0 \
--num_epochs 300 \
--stop_at_epoch 300 \
--batch_size 128 \
--head_tail_accuracy \
--hide_progress \
--data_dir /home/seven/datasets/RAF/ \
--output_dir outputs/raf_res50_experiment/ \
')


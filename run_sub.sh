#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

# Run after:
#   conda activate aaclip

TRAIN_DATASET="mvtec"
TEST_DATASET="visa"
LEVELS=(18 21 24)
EPOCHS=2
TEST_FROM_EPOCH=2
TEST_TO_EPOCH=2

CHECKPOINT="${TRAIN_DATASET}/subspace/18_21_24"

# python train.py \
# 	--dataset "${TRAIN_DATASET}" \
# 	--save_path "ckpt/${CHECKPOINT}" \
# 	--epochs "${EPOCHS}" \
# 	--batch_size 4 \
# 	--num_workers 4 \
# 	--levels "${LEVELS[@]}" \
# 	--proj_dim 768 \
# 	--subspace_dim 30 \
# 	--normal_topk 5 \
# 	--normal_temperature 0.07 \
# 	--normal_bank_source normal_images \
# 	--bank_reduce reservoir \
# 	--bank_candidates_per_class 20000 \
# 	--bank_anchors_per_class 4000 \
# 	--reference_policy all \
# 	--rec_weight 0.0 

for i in $(seq "${TEST_FROM_EPOCH}" "${TEST_TO_EPOCH}")
do
	echo "Testing epoch ${i}..."
	python test.py \
		--dataset "${TEST_DATASET}" \
		--checkpoint /home/ben/VLM/ReSubAD/ckpt/mvtec/ablations/subspace_dim/epoch_2.pth \
		--save_path "ckpt/${CHECKPOINT}/${TEST_DATASET}_test" \
		--save_metrics "ckpt/${CHECKPOINT}/${TEST_DATASET}_test/epoch_${i}_metrics.json" \
		--levels "${LEVELS[@]}" \
		--batch_size 4 \
		--num_workers 4 \
		--pixel_weight 0.8
done
--checkpoint "ckpt/${CHECKPOINT}/epoch_${i}.pth" \
# TEST_DATASETS=("btad" "ksdd2" "dtd")
# TEST_DATASETS=("clinicdb" "colondb" "endo" "isic" "brain_mri" "br35h" "headct" "his")
# for TEST_DATASET in "${TEST_DATASETS[@]}"
# do
# 	echo "Testing dataset ${TEST_DATASET}..."
# 	python test.py \
# 		--dataset "${TEST_DATASET}" \
# 		--checkpoint "ckpt/${CHECKPOINT}/epoch_2.pth" \
# 		--save_path "ckpt/${CHECKPOINT}/${TEST_DATASET}_test" \
# 		--save_metrics "ckpt/${CHECKPOINT}/${TEST_DATASET}_test/epoch_2_metrics.json" \
# 		--levels "${LEVELS[@]}" \
# 		--batch_size 4 \
# 		--num_workers 4 \
# 		--metric_image_size 259 \
# 		--pixel_weight 0.8
# done


#Ablation 
# # number of subspace basis
CHECKPOINT="${TRAIN_DATASET}/ablations/subspace_dim"
# TEST_DATASETS=("dtd" "colondb" "endo" "br35h" "headct" "his")
# for TEST_DATASET in "${TEST_DATASETS[@]}"
# do
# python test.py \
#             --dataset "${TEST_DATASET}" \
#             --checkpoint "ckpt/${CHECKPOINT}/epoch_2.pth" \
#             --save_path "ckpt/${CHECKPOINT}/${TEST_DATASET}_test" \
#             --save_metrics "ckpt/${CHECKPOINT}/${TEST_DATASET}_test/subspace_128.json" \
#             --levels "${LEVELS[@]}" \
#             --batch_size 4 \
#             --num_workers 4 \
#             --pixel_weight 0.8
# done
# SUBSPACE_DIMS=(18 48 64 96 128)
# for SUBSPACE_DIM in "${SUBSPACE_DIMS[@]}"
# do
#     python train.py \
#         --dataset "${TRAIN_DATASET}" \
#         --save_path "ckpt/${CHECKPOINT}" \
#         --epochs "${EPOCHS}" \
#         --batch_size 4 \
#         --num_workers 4 \
#         --levels "${LEVELS[@]}" \
#         --proj_dim 768 \
#         --subspace_dim "${SUBSPACE_DIM}" \
#         --normal_topk 5 \
#         --normal_temperature 0.07 \
#         --normal_bank_source normal_images \
#         --bank_reduce reservoir \
#         --bank_candidates_per_class 20000 \
#         --bank_anchors_per_class 1000 \
#         --reference_policy all \
# 		--rec_weight 0.0

#     for i in $(seq "${TEST_FROM_EPOCH}" "${TEST_TO_EPOCH}")
#     do
#         echo "Testing subspace_dim=${SUBSPACE_DIM}, epoch ${i}..."

#         python test.py \
#             --dataset "${TEST_DATASET}" \
#             --checkpoint "ckpt/${CHECKPOINT}/epoch_${i}.pth" \
#             --save_path "ckpt/${CHECKPOINT}/${TEST_DATASET}_test" \
#             --save_metrics "ckpt/${CHECKPOINT}/${TEST_DATASET}_test/subspace_${SUBSPACE_DIM}.json" \
#             --levels "${LEVELS[@]}" \
#             --batch_size 4 \
#             --num_workers 4 \
#             --pixel_weight 0.8
#     done
# done


# memory bank size
# CHECKPOINT="${TRAIN_DATASET}/ablations/mb_size"
# MB_SIZES=(500)
# for MB_SIZE in "${MB_SIZES[@]}"
# do
#     python train.py \
#         --dataset "${TRAIN_DATASET}" \
#         --save_path "ckpt/${CHECKPOINT}" \
#         --epochs "${EPOCHS}" \
#         --batch_size 4 \
#         --num_workers 4 \
#         --levels "${LEVELS[@]}" \
#         --proj_dim 768 \
#         --subspace_dim 48 \
#         --normal_topk 10 \
#         --normal_temperature 0.07 \
#         --normal_bank_source normal_images \
#         --bank_reduce reservoir \
#         --bank_candidates_per_class 10000 \
#         --bank_anchors_per_class "${MB_SIZE}" \
#         --reference_policy all \
# 		# --rec_weight 0.1

#     for i in $(seq "${TEST_FROM_EPOCH}" "${TEST_TO_EPOCH}")
#     do
#         echo "Testing mb_size=${MB_SIZE}, epoch ${i}..."

#         python test.py \
#             --dataset "${TEST_DATASET}" \
#             --checkpoint "ckpt/${CHECKPOINT}/epoch_${i}.pth" \
#             --save_path "ckpt/${CHECKPOINT}/${TEST_DATASET}_test" \
#             --save_metrics "ckpt/${CHECKPOINT}/${TEST_DATASET}_test/mb_${MB_SIZE}.json" \
#             --levels "${LEVELS[@]}" \
#             --batch_size 4 \
#             --num_workers 4 \
#             --pixel_weight 0.8
#     done

# 	rm ckpt/${CHECKPOINT}/memory_bank.pth
# done

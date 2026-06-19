#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

# Run after:
#   conda activate aaclip

TRAIN_DATASET="visa"
TEST_DATASET="mvtec"
LEVELS=(18 21 24)
EPOCHS=2
TEST_FROM_EPOCH=2
TEST_TO_EPOCH=2

CHECKPOINT="${TRAIN_DATASET}/subspace/18_21_24"

python train.py \
	--dataset "${TRAIN_DATASET}" \
	--save_path "ckpt/${CHECKPOINT}" \
	--epochs "${EPOCHS}" \
	--batch_size 4 \
	--num_workers 4 \
	--levels "${LEVELS[@]}" \
	--proj_dim 768 \
	--subspace_dim 30 \
	--normal_topk 5 \
	--normal_temperature 0.07 \
	--normal_bank_source normal_images \
	--bank_reduce reservoir \
	--bank_candidates_per_class 10000 \
	--bank_anchors_per_class 2000 \
	--reference_policy all \
	--rec_weight 0.0

for i in $(seq "${TEST_FROM_EPOCH}" "${TEST_TO_EPOCH}")
do
	echo "Testing epoch ${i}..."
	python test.py \
		--dataset "${TEST_DATASET}" \
		--checkpoint "ckpt/${CHECKPOINT}/epoch_${i}.pth" \
		--save_path "ckpt/${CHECKPOINT}/${TEST_DATASET}_test" \
		--save_metrics "ckpt/${CHECKPOINT}/${TEST_DATASET}_test/epoch_${i}_metrics.json" \
		--levels "${LEVELS[@]}" \
		--batch_size 4 \
		--num_workers 4 \
		--pixel_weight 0.8
done

TEST_DATASETS=("btad" "ksdd2" "dtd")
# TEST_DATASETS=("brain_mri" "br35h" "headct" "clinicdb" "colondb" "endo")
# TEST_DATASETS=("dagm" "kvasir")
TEST_DATASETS=("isic")
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
# 		--metric_image_size 518 \
# 		--pixel_weight 0.8
# done

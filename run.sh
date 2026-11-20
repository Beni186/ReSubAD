#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

TRAIN_DATASET="visa"
TEST_DATASET="mvtec"
LEVELS=(18 21 24)
EPOCHS=2
TEST_FROM_EPOCH=2
TEST_TO_EPOCH=2

CHECKPOINT="${TRAIN_DATASET}"

python train.py \
	--dataset "${TRAIN_DATASET}" \
	--save_path "ckpt/${CHECKPOINT}" \
	--levels "${LEVELS[@]}" \
	--epochs "${EPOCHS}" \
	--batch_size 4 \
	--num_workers 4 

for i in $(seq "${TEST_FROM_EPOCH}" "${TEST_TO_EPOCH}")
do
	echo "Testing epoch ${i}..."
	python test.py \
		--dataset "${TEST_DATASET}" \
		--checkpoint "ckpt/${CHECKPOINT}/epoch_${i}.pth" \
		--save_path "ckpt/${CHECKPOINT}/${TEST_DATASET}_test" \
		--levels "${LEVELS[@]}" \
		--batch_size 4 \
		--num_workers 4 \
done
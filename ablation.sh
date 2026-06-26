TRAIN_DATASET="mvtec"
TEST_DATASET="visa"
LEVELS=(18 21 24)

CHECKPOINT="${TRAIN_DATASET}/subspace/18_21_24"

SUBSPACE_DIMS=(24 36 48)
for SUBSPACE_DIM in "${SUBSPACE_DIMS[@]}"
do

python analysis/no_nrr/train.py --dataset ${TRAIN_DATASET} --epochs 2 --batch_size 4 \
  --proj_dim 768 --subspace_dim ${SUBSPACE_DIM} --save_path analysis/no_nrr/ckpt/${TRAIN_DATASET}


python analysis/no_nrr/test.py --dataset ${TEST_DATASET} --pixel_weight 0.8 \
  --checkpoint analysis/no_nrr/ckpt/${TRAIN_DATASET}/epoch_2.pth \
  --save_metrics analysis/no_nrr/ckpt/${TRAIN_DATASET}/${TEST_DATASET}.json --save_path analysis/no_nrr/ckpt/${TRAIN_DATASET}/${TEST_DATASET}

done

SUBSPACE_DIMS=(18 24 30 36 48)
for SUBSPACE_DIM in "${SUBSPACE_DIMS[@]}"
do

python analysis/no_nrr/train.py --dataset ${TRAIN_DATASET} --epochs 2 --batch_size 4 \
  --proj_dim 512 --subspace_dim ${SUBSPACE_DIM} --save_path analysis/no_nrr/ckpt/${TRAIN_DATASET}


python analysis/no_nrr/test.py --dataset ${TEST_DATASET} --pixel_weight 0.8 \
  --checkpoint analysis/no_nrr/ckpt/${TRAIN_DATASET}/epoch_2.pth \
  --save_metrics analysis/no_nrr/ckpt/${TRAIN_DATASET}/${TEST_DATASET}.json --save_path analysis/no_nrr/ckpt/${TRAIN_DATASET}/${TEST_DATASET}

done

TRAIN_DATASET="visa"
TEST_DATASET="mvtec"

SUBSPACE_DIMS=(18 24 36 48)
for SUBSPACE_DIM in "${SUBSPACE_DIMS[@]}"
do

python analysis/no_nrr/train.py --dataset ${TRAIN_DATASET} --epochs 2 --batch_size 4 \
  --proj_dim 768 --subspace_dim ${SUBSPACE_DIM} --save_path analysis/no_nrr/ckpt/${TRAIN_DATASET}


python analysis/no_nrr/test.py --dataset ${TEST_DATASET} --pixel_weight 0.8 \
  --checkpoint analysis/no_nrr/ckpt/${TRAIN_DATASET}/epoch_2.pth \
  --save_metrics analysis/no_nrr/ckpt/${TRAIN_DATASET}/${TEST_DATASET}.json --save_path analysis/no_nrr/ckpt/${TRAIN_DATASET}/${TEST_DATASET}

done

SUBSPACE_DIMS=(18 24 30 36 48)
for SUBSPACE_DIM in "${SUBSPACE_DIMS[@]}"
do

python analysis/no_nrr/train.py --dataset ${TRAIN_DATASET} --epochs 2 --batch_size 4 \
  --proj_dim 512 --subspace_dim ${SUBSPACE_DIM} --save_path analysis/no_nrr/ckpt/${TRAIN_DATASET}


python analysis/no_nrr/test.py --dataset ${TEST_DATASET} --pixel_weight 0.8 \
  --checkpoint analysis/no_nrr/ckpt/${TRAIN_DATASET}/epoch_2.pth \
  --save_metrics analysis/no_nrr/ckpt/${TRAIN_DATASET}/${TEST_DATASET}.json --save_path analysis/no_nrr/ckpt/${TRAIN_DATASET}/${TEST_DATASET}

done
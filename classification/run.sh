MODEL=resnet18
DATASETS=(
	dermamnist
	tissuemnist
)
BITS=(
	2
	3
	4
)

for DATASET in ${DATASETS[@]}
do
    for BIT in ${BITS[@]}
    do
        echo "Testing $DATASET with W${BIT}A${BIT} ..."
        python uniform_test.py \
			--seed=0 \
            --dataset=$DATASET \
			--model=$MODEL \
			--pretrained=./checkpoints/${MODEL}_${DATASET}.pth \
            --weight_bit=$BIT \
            --act_bit=$BIT \
            --batch_size=64 \
            --test_batch_size=512 \
            --init_data_path=/home/dataset/imagenet
    done
done

MODEL=resnet18

for DATASET in dermamnist tissuemnist
do
    for BIT in 2 3 4
    do
        echo "Testing $DATASET with W${BIT}A${BIT} ..."
        python uniform_test.py \
            --dataset=$DATASET \
			--model=$MODEL \
			--pretrained=./checkpoints/${MODEL}_${DATASET}.pth \
            --weight_bit=$BIT \
            --act_bit=$BIT \
            --batch_size=64 \
            --test_batch_size=512
    done
done

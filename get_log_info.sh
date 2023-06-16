valdatasets=(
    imagenet
    omniglot
    aircraft
    cub
    dtd
    quickdraw
    fungi
    vgg
    mscoco
)

grep "Loss" $1

for dataset in "${valdatasets[@]}"; do
    echo "Performance of $dataset"
    grep "$dataset 5nshots" $1
done

grep "]5nshots" $1
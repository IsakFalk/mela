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

grep "Loss" out.log

for dataset in "${valdatasets[@]}"; do
    echo "Performance of $dataset"
    grep "$dataset 20nshots" out.log
done

grep "^20nshots" out.log
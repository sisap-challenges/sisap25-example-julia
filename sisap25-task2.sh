echo "RUN AS: bash -x task2.sh 2>&1 | tee log-task2.txt"

PATH_TO_HOST_DIR=/home/sisap23evaluation/data2025/without-gold
## WARNING RUN WITHOUT NETWORK
PATH_TO_CONTAINER_DIR=/sisap25/data
OUT_PATH_TO_HOST_DIR=$(pwd)/results-task2
OUT_PATH_TO_CONTAINER_DIR=/sisap25/results

mkdir $OUT_PATH_TO_HOST_DIR
echo "==== pwd: $(pwd)"
echo "==== directory listing: "
ls
echo "==== environment"
set
echo "==== RUN BEGINS $(date)"
docker run \
    -it \
    --cpus=8 \
    --memory=16g \
    --memory-swap=16g \
    --memory-swappiness 0 \
    --network none \
    --volume $PATH_TO_HOST_DIR:$PATH_TO_CONTAINER_DIR:ro \
    --volume $OUT_PATH_TO_HOST_DIR:$OUT_PATH_TO_CONTAINER_DIR:rw \
    sisap25/similaritysearch julia --project=. -O3 -Cnative -t8 -L sisap2025.jl -e 'main_task2(; preprocessing="pca-i8", maxoutdim=160, minrecall=0.9, minrecallsearch=0.9, logbase=1.3)'


echo "==== RUN ENDS $(date)"



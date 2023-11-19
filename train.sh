export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=1

export EXP=Col-xlmr-ZS-max180
export BASE_MODEL=xlm-roberta-base

export TRIPLET=/mnt/hdd4/jhkim980112/Dataset/msmarco/triples.train.small.shuffled.tsv
#/home/jhkim980112/workspace/dataset/msmarco/qidpidtriples.rnd-shuf.train.tsv
export COLLECTION=/home/jhkim980112/workspace/dataset/msmarco/collection.tsv
export QUERY=/home/jhkim980112/workspace/dataset/msmarco/queries.train.tsv

#--collection $COLLECTION --queries $QUERY \

CUDA_VISIBLE_DEVICES="3" \
    python -m colbert.train \
    --base_model $BASE_MODEL \
    --triples $TRIPLET \
    --amp --doc_maxlen 180 --bsize 32 --accum 1 --maxsteps 200_000 --similarity l2 \
    --root ./experiments/ --experiment $EXP --run mMARCO_en-en
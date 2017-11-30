#! /bin/bash
export model_path="/home/kuznetsov/tmp/thread35"
# export generator_path="/home/kuznetsov/embeddings/fasttext/models/lenta+wiki+ted+tedx/lenta+wiki+ted+tedx-0.bin"
# rm -rf ${model_path}/model
mkdir  ${model_path}/model
cp -v $0 ${model_path}/model
pwd > ${model_path}/model/executed_path.meta


python -m nmt.nmt     \
  --src=cor --tgt=man     \
  --vocab_prefix=${model_path}/data/vocab     \
  --train_prefix=${model_path}/data/train     \
  --dev_prefix=${model_path}/data/dev_test     \
  --test_prefix=${model_path}/data/test     \
  --out_dir=${model_path}/model     \
  --batch_size 128     \
  --num_train_steps=3000000     \
  --steps_per_stats=100     \
  --num_layers=2     \
  --num_units=1024     \
  --dropout=0.2     \
  --src_max_len=60    \
  --tgt_max_len=30  \
  --metrics=bleu

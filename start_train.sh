#! /bin/bash

export model_path="/home/kuznetsov/tmp/nmt_for_test"
rm -rf ${model_path}/model
mkdir  ${model_path}/model
python -m nmt.nmt     \
  --src=cor --tgt=man     \
  --pretrain_dec_emb_path=${model_path}/data/lenta_embed.emb \
  --vocab_prefix=${model_path}/data/lenta_vocab     \
  --train_prefix=${model_path}/data/train     \
  --dev_prefix=${model_path}/data/dev_test     \
  --test_prefix=${model_path}/data/test     \
  --out_dir=${model_path}/model     \
  --batch_size 128     \
  --num_train_steps=3000000     \
  --steps_per_stats=100     \
  --num_layers=2     \
  --num_units=512     \
  --dropout=0.2     \
  --src_max_len=250    \
  --tgt_max_len=130    \
  --attention=luong     \
  --metrics=bleu

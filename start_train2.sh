#! /bin/bash

export model_path="/home/kuznetsov/tmp/nmt_dev2"
# rm -rf ${model_path}/models
# mkdir  ${model_path}/models
  python -m nmt.nmt     \
    --src=cor --tgt=man     \
    --pretrain_enc_emb_path=${model_path}/data/lenta_embed.emb \
    --embedding_generator_path="/home/kuznetsov/embeddings/fastText/models/lenta/embeddings_lenta_0.1.bin" \
    --vocab_prefix=${model_path}/data/vocab     \
    --train_prefix=${model_path}/data/train     \
    --dev_prefix=${model_path}/data/dev_test     \
    --test_prefix=${model_path}/data/test     \
    --out_dir=${model_path}/model     \
    --encoder_type bi \
    --batch_size 128     \
    --num_train_steps=3000000     \
    --steps_per_stats=100     \
    --num_layers=2     \
    --num_units=100     \
    --dropout=0.2     \
    --src_max_len=250    \
    --tgt_max_len=130    \
    --attention=luong     \
    --metrics=bleu
##\ #  --pretrain_enc_emb_path=${model_path}/data/lenta_embed.emb \

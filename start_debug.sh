#! /bin/bash

export model_path="/home/kuznetsov/tmp/thread17"
export generator_path="/home/kuznetsov/embeddings/fasttext/models/model-l+w+2t-1/lenta+wiki+ted+tedx-1.bin"
rm -rf ${model_path}/model
mkdir  ${model_path}/model
#--pretrain_dec_emb_path=${model_path}/data/enc_embeddings.emb \
#--pretrain_dec_emb_path=${model_path}/data/dec_embeddings.emb \




python -m nmt.nmt     \
  --src=cor --tgt=man     \
  --embedding_generator_path=${generator_path} \
  --pretrain_enc_emb_path=${model_path}/data/enc_embeddings.emb \
  --pretrain_dec_emb_path=${model_path}/data/enc_embeddings.emb \
  --share_vocab True \
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
  --num_units=1024     \
  --dropout=0.2     \
  --src_max_len=140    \
  --tgt_max_len=140  \
  --attention=luong     \
  --metrics=bleu
# #! /bin/bash
#
# export model_path="/home/kuznetsov/tmp/thread17"
# export generator_path="/home/kuznetsov/embeddings/fasttext/models/model-l+w+2t-1/lenta+wiki+ted+tedx-1.bin"
# rm -rf ${model_path}/model
# mkdir  ${model_path}/model
# #--pretrain_enc_emb_path=${model_path}/data/enc_embeddings.emb \
# # --share_vocab True \
# #--pretrain_dec_emb_path=${model_path}/data/dec_embeddings.emb \
# # --attention=luong     \
# # --log_device_placement True \
# # --num_gpus 3 \
#
#
# python -m nmt.nmt     \
#   --src=cor --tgt=man     \
#   --embedding_generator_path=${generator_path} \
#   --vocab_prefix=${model_path}/data/vocab     \
#   --train_prefix=${model_path}/data/train     \
#   --dev_prefix=${model_path}/data/dev_test     \
#   --test_prefix=${model_path}/data/test     \
#   --out_dir=${model_path}/model     \
#   --num_gpus 2 \
#   --encoder_type bi \
#   --batch_size 128     \
#   --num_train_steps=3000000     \
#   --steps_per_stats=100     \
#   --num_layers=2     \
#   --num_units=128     \
#   --dropout=0.2     \
#   --src_max_len=140    \
#   --tgt_max_len=140  \
#   --metrics=bleu
# #
#
# python -m nmt.nmt     \
#   --src=cor --tgt=man     \
#   --embedding_generator_path=${generator_path} \
#   --pretrain_enc_emb_path=${model_path}/data/enc_embeddings.emb \
#   --pretrain_dec_emb_path=${model_path}/data/enc_embeddings.emb \
#   --vocab_prefix=${model_path}/data/vocab     \
#   --train_prefix=${model_path}/data/train     \
#   --dev_prefix=${model_path}/data/dev_test     \
#   --test_prefix=${model_path}/data/test     \
#   --out_dir=${model_path}/model     \
#   --encoder_type bi \
#   --batch_size 20     \
#   --num_train_steps=3000000    \
#   --steps_per_stats=10     \
#   --num_layers=2     \
#   --num_units=256     \
#   --dropout=0.2     \
#   --src_max_len=250    \
#   --tgt_max_len=250  \
#   --attention=luong     \
#   --metrics=bleu

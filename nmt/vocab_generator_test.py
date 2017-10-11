# from nmt.vocab_generator import VocabGeneration
from vocab_generator import VocabGeneration

import os
#change
model_path = "/home/kuznetsov/embeddings/fastText/models/lenta/embeddings_lenta_0.1.bin"
# model_path = "/home/kuznetsov/embeddings/fastText/model.bin"
root_dir = "/home/kuznetsov/embeddings/fastText/models"

input_voc = os.path.join(root_dir, "lenta_vocab.voc")
input_embedfile = os.path.join(root_dir, "lenta_embed.emb")
input_vocab = os.path.join(root_dir, "lenta_vocab1.emb")
# voc_gen = VocabGeneration(model_path)
#
# voc_gen.load_vocab_from_file(input_voc)
#
# emb, miss_v, emb_v = voc_gen.get_all_data()
# print(len(emb))
# print(len(miss_v))
# print(len(emb_v))
# print((emb))
# print((miss_v))
# print((emb_v))
# print(emb)
#
# voc_gen.save_embeddings()
voc_gen1 = VocabGeneration(model_path)
voc_gen1.load_vocab_from_corpus_file(input_voc, tag_list =["<unk>", "<s>", "</s>"])

emb1, miss_v1, emb_v1 = voc_gen1.get_all_data()
miss_v1
emb_v1[:20]
emb1
voc_gen1.save_embeddings(input_embedfile)
voc_gen1.save_embedded_vocab(input_vocab)

voc_gen1._

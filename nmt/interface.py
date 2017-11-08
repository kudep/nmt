# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TensorFlow NMT model implementation."""
from __future__ import print_function

import argparse
import os
import random
import sys
import re

# import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

from . import inference
from . import train
from . import apply_bpe
from .utils import evaluation_utils
from .utils import misc_utils as utils
from .utils import vocab_utils
from .nmt import *
from .embeddings_generator import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
tf.logging.set_verbosity(tf.logging.ERROR)



#interface section

class AgentMemory():
    def __init__(self,context_file):
        self.man_context = []
        self.cor_context = []
        self.context = []
        self.context_file = context_file
    def get_memory(self):
        return self.cor_context, self.man_context, self.context, self.context_file

class Agents():
    def __init__(self, model_dir, ckpt_name = None, best_bleu = False, context_len=5, man_context_len=1, verbose_output = False, embedding_generator_path=None, bpe_enc=None, bpe_dec=None):
        #Files
        assert not model_dir is None
        buf_path = "/tmp/nmt_chit-chat/buffers/"
        self.inf_input_file = os.path.join(buf_path, 'inference_input_buf')
        self.inf_output_file = os.path.join(buf_path, 'inference_output_buf')
        self.base_context_file = os.path.join(buf_path, 'context')

        model_gen = os.path.join(model_dir,'gen')
        bpe_codes_file_path = os.path.join(model_dir,'data/codes.txt')
        bpe_voc_file_path = os.path.join(model_dir,'data/bpeshare.voc')
        model_dir = os.path.join(model_dir,'model')
        self._mkprotecteddir(buf_path)
        nmt_parser = argparse.ArgumentParser()
        add_arguments(nmt_parser)
        ckpt_file = self._ckpt_select(model_dir,ckpt_name, best_bleu)
        self.embed_gen_flag = False
        self.embeddings_generator = None
        if embedding_generator_path:
            print("Инициализация генератора эмбеддингов, будет использован файл {}".format(embedding_generator_path))
            # print("Attention! Not initialize embeddings model.")
            self.embeddings_generator = EmbeddingsGenerator(embedding_generator_path)
            self.embed_gen_flag = True
            self._mkprotecteddir(model_gen)
            self.pretrain_enc_emb_path = os.path.join(model_gen, "encoder_embeddings.emb")
            # pretrain_dec_emb_path = "./gen/decoder_embeddings.emb"
            self.src_vocab_file = os.path.join(model_gen, "src.voc")
            # tgt_vocab_file = "./gen/tgt.voc"

        print("Будет использована модель {}".format(ckpt_file))

        self._ext_add_options(nmt_parser,model_dir,ckpt_file, verbose_output)
        flags, unparsed = nmt_parser.parse_known_args()

        # Insert in class
        flags = flags # Unthinking
        if bpe_enc or bpe_dec:
            self.bpe_enc = bpe_enc
            self.bpe_dec = bpe_dec
        else:
            self.bpe_enc = False
            self.bpe_dec = False

        if flags.bpe_delimiter:
            self.bpe_enc = True
            self.bpe_dec = True

        if bpe_enc or bpe_dec:
            #Separator for bpe
            self.separator='@@'
            self.apply_bpe = apply_bpe.apply_bpe(bpe_codes_file_path, bpe_voc_file_path, separator=self.separator)
        default_hparams = create_hparams(flags)
        train_fn =  train.train
        inference_fn = inference.inference
        self.nmt_model, self.nmt_arg = preparation_for_inference(flags,
                                        default_hparams, train_fn, inference_fn,
                                        embedding_generation = self.embed_gen_flag)

        self.model_dir =  model_dir
        self.context_len = context_len
        self.man_context_len = man_context_len
        self.agents_memory = dict()



    def deploy_agent(self, agent_id = 0, reset = True):
        if self.agents_memory.get(agent_id, None):
            if reset: self.reset_agent(agent_id)
        else:
            self.agents_memory[agent_id] = AgentMemory(self.base_context_file +'-' +type(agent_id).__name__+ '-' + str(agent_id)+'.buf')

    def reset_agent(self, agent_id = 0):
        if self.agents_memory.get(agent_id, None):
            self.agents_memory[agent_id] = AgentMemory(self.base_context_file +'-' +type(agent_id).__name__+ '-' + str(agent_id)+'.buf')

    def send(self, msg = '', agent_id = 0):
        #Update context
        self.deploy_agent(agent_id = agent_id, reset = False) # Create agent if agent not exist
        # man_start_tag = " <MAN_START> "
        # cor_start_tag = " <COR_START> "
        man_start_tag = " "
        cor_start_tag = " "
        line = re.sub(' +', ' ', cor_start_tag + self._preproc(msg))
        self._change_agent_context(line, agent_id, "human")
        _, _, context, _ = self.agents_memory[agent_id].get_memory()
        #Update work files
        if self.embed_gen_flag:
            self.embeddings_generator.load_vocab_from_list_of_rows(context,
                                        tag_list = ["<unk>", "<s>", "</s>"])
            # print("Attention! Not save generated embeddings.")
            self.embeddings_generator.save_embeddings(self.pretrain_enc_emb_path)
            self.embeddings_generator.save_embedded_vocab(self.src_vocab_file)
        self._write_into_file(context, self.inf_input_file)
        #Start model for generation
        self.nmt_model(*self.nmt_arg)
        answer = self._read_from_file(self.inf_output_file)
        #Save answer
        line = re.sub(' +', ' ', man_start_tag + self._preproc(answer))
        self._change_agent_context(line, agent_id, "model")
        _, _, context, context_file = self.agents_memory[agent_id].get_memory()
        self._context_into_buf(context, context_file)

        if self.bpe_dec:
            pattern = '({} )'.format(self.separator)
            answer = re.sub(pattern, '', answer)
        return answer

    def get_content(self, agent_id = 0):
        _, _, context, _ = self.agents_memory[agent_id].get_memory()
        return context

    def interactive_dialogue(self, prompt = 'Вы: ', agent_id = 0):
        while True:
            answer = self.send(input(prompt), agent_id)
            print("NMT: " + answer)

    def _context_into_buf(self, context, context_file):
        self._write_into_file(context,context_file,'\n')

    def _mkprotecteddir(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def _write_into_file(self, context,path, addition = ""):
        with open(path, "wt") as f:
            for line in context:
                f.write(line+addition )

    def _read_from_file(self, path):
        with open(path, "r") as pipe:
            line = pipe.read()
            return line

    def _ext_add_options(self, parser,model_dir,ckpt, verbose_output):
        parser.set_defaults(out_dir = model_dir)
        parser.set_defaults(ckpt = ckpt)
        parser.set_defaults(inference_output_file = self.inf_output_file)
        parser.set_defaults(inference_input_file = self.inf_input_file)
        parser.set_defaults(verbose_output = False)

    def _preproc(self, line):
        line = re.sub(r'[\s+]', ' ', line)
        line = re.sub(r'(\\n)', ' ', line)
        line = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", line)
        line = re.sub(r"([\*\"\'\\\/\|\{\}\[\]\;\:\<\>\,\.\?\*\(\)])", r" \1 ", line)
        line = re.sub(' +', ' ', line)
        if self.bpe_enc:
            line = self.apply_bpe(line)
        return line
    def _change_agent_context(self, line, agent_id = 0, speaker=None):
        cor_context, man_context, context, _ = self.agents_memory[agent_id].get_memory()
        context_len, man_context_len = self.context_len, self.man_context_len
        context = []
        if speaker=="model":
            man_context.insert(0,line)
        else:
            assert speaker=="human"
            cor_context.insert(0,line)
        assert context_len > man_context_len
        half_context_len = context_len//2

        man_iter = min(len(man_context), man_context_len, man_context_len)
        cor_iter = min(len(cor_context), context_len, context_len - man_context_len)

        share_iter_max = min(man_iter,cor_iter)
        addition_iter_max = max(man_iter,cor_iter)

        excluding_man = man_iter > cor_iter

        man_context = man_context[0:man_iter]
        cor_context = cor_context[0:cor_iter]

        #Share filling
        # man_context.reverse()
        for idx in range(share_iter_max):
            context.append(cor_context[idx])
            context.append(man_context[idx])
        # man_context.reverse()

        #Corparete filling
        for idx in range(share_iter_max,addition_iter_max,1):
            if excluding_man:
                context.append(man_context[idx])
            else:
                context.append(cor_context[idx])
        context.reverse()
        self.agents_memory[agent_id].cor_context = cor_context
        self.agents_memory[agent_id].man_context = man_context
        self.agents_memory[agent_id].context = context

    def _ckpt_select(self, model_dir, ckpt_name = None, best_bleu = False):
        # ckpt_name = os.path.join(model_dir, 'translate.ckpt-98000')
        def get_checkp_from_file(path):
            ckpt_name=None
            with open(path) as f:
                 ckpt_name = f.readline().split('/')[-1].split('"')[0]
            return ckpt_name
        if ckpt_name:
            model_path = os.path.join(model_dir, ckpt_name)
        else:
            if best_bleu:
                checkpoint = os.path.join(model_dir,'best_bleu','checkpoint')
                ckpt_name = get_checkp_from_file(checkpoint)
                model_path = os.path.join(model_dir,'best_bleu',ckpt_name)
            else:
                checkpoint = os.path.join(model_dir,'checkpoint')
                ckpt_name= get_checkp_from_file(checkpoint)
                model_path = os.path.join(model_dir,ckpt_name)
        return model_path

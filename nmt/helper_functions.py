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

"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


from tensorflow.contrib.seq2seq.python.ops import decoder
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.distributions import bernoulli
from tensorflow.python.ops.distributions import categorical
from tensorflow.python.util import nest
from . import model_helper
import tensorflow as tf
__all__ = ["EmbedHelper"]

def _unstack_ta(inp):
  return tensor_array_ops.TensorArray(
      dtype=inp.dtype, size=array_ops.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)

class EmbedHelper(tf.contrib.seq2seq.TrainingHelper):
    # pass
  # def __init__(self, inputs, sequence_length, time_major=False, name=None):
  #   with ops.name_scope(name, "EmbebHelper", [inputs, sequence_length]):
  #     inputs = ops.convert_to_tensor(inputs, name="inputs")
  #     if not time_major:
  #       inputs = nest.map_structure(_transpose_batch_time, inputs)
  #
  #     self._input_tas = nest.map_structure(_unstack_ta, inputs)
  #     self._sequence_length = ops.convert_to_tensor(
  #         sequence_length, name="sequence_length")
  #     if self._sequence_length.get_shape().ndims != 1:
  #       raise ValueError(
  #           "Expected sequence_length to be a vector, but received shape: %s" %
  #           self._sequence_length.get_shape())
  #
  #     self._zero_inputs = nest.map_structure(
  #         lambda inp: array_ops.zeros_like(inp[0, :]), inputs)
  #
  #     self._batch_size = array_ops.size(sequence_length)
  #
  # def initialize(self, name=None):
  #     with ops.name_scope(name, "EmbebHelperInitialize"):
  #       finished = math_ops.equal(0, self._sequence_length)
  #       all_finished = math_ops.reduce_all(finished)
  #       next_inputs = control_flow_ops.cond(
  #           all_finished, lambda: self._zero_inputs,
  #           lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
  #       return (finished, next_inputs)
  #
  def sample(self, time, outputs, name=None, **unused_kwargs):
      with ops.name_scope(name, "EmbebHelperSample", [time, outputs]):
        sample = outputs[-1]
        return sample
  #
  # def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
  #     """next_inputs_fn for EmbebHelper."""
  #     with ops.name_scope(name, "EmbebHelperNextInputs",
  #                         [time, outputs, state]):
  #       next_time = time + 1
  #       finished = (next_time >= self._sequence_length)
  #       all_finished = math_ops.reduce_all(finished)
  #       def read_from_ta(inp):
  #         return inp.read(next_time)
  #       next_inputs = control_flow_ops.cond(
  #           all_finished, lambda: self._zero_inputs,
  #           lambda: nest.map_structure(read_from_ta, self._input_tas))
  #       return (finished, next_inputs, state)

class ExtendGreedyEmbeddingHelper(tf.contrib.seq2seq.GreedyEmbeddingHelper):
  def __init__(self, embedding, start_tokens, end_token, out_embedding_w, out_embedding_b):
      super(ExtendGreedyEmbeddingHelper, self).__init__(embedding, start_tokens, end_token)
      self.embedding = embedding
      self.out_embedding_w = out_embedding_w
      self.out_embedding_b = out_embedding_b

  def sample(self, time, outputs, state, name=None):
    """sample for GreedyEmbeddingHelper."""
    del time, state  # unused by sample_fn
    # Outputs are logits, use argmax to get the most probable id
    if not isinstance(outputs, ops.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))
    print('outputs {}'.format(outputs))
    rnn_outputs = tf.tensordot(outputs, self.out_embedding_w, axes =[[-1],[0]]) + tf.reshape(self.out_embedding_b, [1,1,-1])
    print('embedding {}'.format(self.embedding))
    print('rnn_outputs {}'.format(rnn_outputs))
    sub = tf.subtract(self.embedding,rnn_outputs)
    square_sum = tf.reduce_sum(tf.square(sub), -1)
    euclid_dist = tf.sqrt(tf.convert_to_tensor(square_sum, dtype=tf.float32))
    print('euclid_dist {}'.format(euclid_dist))
    sample_ids = math_ops.cast(
        tf.argmin(euclid_dist,-1), dtypes.int32)
    return sample_ids

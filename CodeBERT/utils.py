# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from sklearn.metrics import f1_score

csv.field_size_limit(2**31-1)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data. Modified for bi-encoder architecture with separate code/query processing"""
    def __init__(self, 
                 code_input_ids, code_input_mask, code_segment_ids,
                 query_input_ids, query_input_mask, query_segment_ids,
                 label_id):
        self.code_input_ids = code_input_ids
        self.code_input_mask = code_input_mask
        self.code_segment_ids = code_segment_ids
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                lines.append(line)
            return lines


class CodesearchProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir, train_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, train_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, train_file)), "train")

    def get_dev_examples(self, data_dir, dev_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, dev_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, dev_file)), "dev")

    def get_test_examples(self, data_dir, test_file):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, test_file)))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, test_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            if (set_type == 'test'):
                label = self.get_labels()[0]
            else:
                label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        if (set_type == 'test'):
            return examples, lines
        else:
            return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, debug=False,):
    """ Modified for bi-encoder architecture with separate code/query processing """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if debug and ex_index > 10000:
            break

        # Process CODE input (text_a)
        code_tokens = tokenizer.tokenize(example.text_a)
        # Truncate code to max length accounting for [CLS] and [SEP]
        if len(code_tokens) > max_seq_length - 2:
            code_tokens = code_tokens[:(max_seq_length - 2)]
        
        # Add special tokens and create code features
        code_tokens = code_tokens + [sep_token]
        code_segment_ids = [sequence_a_segment_id] * len(code_tokens)
        if cls_token_at_end:
            code_tokens = code_tokens + [cls_token]
            code_segment_ids = code_segment_ids + [cls_token_segment_id]
        else:
            code_tokens = [cls_token] + code_tokens
            code_segment_ids = [cls_token_segment_id] + code_segment_ids

        code_input_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        code_input_mask = [1 if mask_padding_with_zero else 0] * len(code_input_ids)

        # Pad code sequence
        padding_length = max_seq_length - len(code_input_ids)
        if pad_on_left:
            code_input_ids = ([pad_token] * padding_length) + code_input_ids
            code_input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + code_input_mask
            code_segment_ids = ([pad_token_segment_id] * padding_length) + code_segment_ids
        else:
            code_input_ids = code_input_ids + ([pad_token] * padding_length)
            code_input_mask = code_input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            code_segment_ids = code_segment_ids + ([pad_token_segment_id] * padding_length)

        # Process QUERY input (text_b)
        query_tokens = tokenizer.tokenize(example.text_b)
        # Truncate query to max length accounting for [CLS] and [SEP]
        if len(query_tokens) > max_seq_length - 2:
            query_tokens = query_tokens[:(max_seq_length - 2)]
        
        # Add special tokens and create query features
        query_tokens = query_tokens + [sep_token]
        query_segment_ids = [sequence_a_segment_id] * len(query_tokens)
        if cls_token_at_end:
            query_tokens = query_tokens + [cls_token]
            query_segment_ids = query_segment_ids + [cls_token_segment_id]
        else:
            query_tokens = [cls_token] + query_tokens
            query_segment_ids = [cls_token_segment_id] + query_segment_ids

        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        query_input_mask = [1 if mask_padding_with_zero else 0] * len(query_input_ids)

        # Pad query sequence
        padding_length = max_seq_length - len(query_input_ids)
        if pad_on_left:
            query_input_ids = ([pad_token] * padding_length) + query_input_ids
            query_input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + query_input_mask
            query_segment_ids = ([pad_token_segment_id] * padding_length) + query_segment_ids
        else:
            query_input_ids = query_input_ids + ([pad_token] * padding_length)
            query_input_mask = query_input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            query_segment_ids = query_segment_ids + ([pad_token_segment_id] * padding_length)

        # Validate lengths
        assert len(code_input_ids) == max_seq_length
        assert len(query_input_ids) == max_seq_length
        assert len(code_input_mask) == max_seq_length
        assert len(query_input_mask) == max_seq_length
        assert len(code_segment_ids) == max_seq_length
        assert len(query_segment_ids) == max_seq_length

        # Label processing
        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        # Example logging
        if ex_index < 5:
            logger.info("*** Bi-Encoder Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("Code tokens: %s" % " ".join([str(x) for x in code_tokens]))
            logger.info("Code input_ids: %s" % " ".join([str(x) for x in code_input_ids]))
            logger.info("Code input_mask: %s" % " ".join([str(x) for x in code_input_mask]))
            logger.info("Code segment_ids: %s" % " ".join([str(x) for x in code_segment_ids]))
            logger.info("Query tokens: %s" % " ".join([str(x) for x in query_tokens]))
            logger.info("Query input_ids: %s" % " ".join([str(x) for x in query_input_ids]))
            logger.info("Query input_mask: %s" % " ".join([str(x) for x in query_input_mask]))
            logger.info("Query segment_ids: %s" % " ".join([str(x) for x in query_segment_ids]))
            logger.info("Label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(code_input_ids=code_input_ids,
                          code_input_mask=code_input_mask,
                          code_segment_ids=code_segment_ids,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          label_id=label_id))
    return features



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "codesearch":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


processors = {
    "codesearch": CodesearchProcessor,
}

output_modes = {
    "codesearch": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "codesearch": 2,
}

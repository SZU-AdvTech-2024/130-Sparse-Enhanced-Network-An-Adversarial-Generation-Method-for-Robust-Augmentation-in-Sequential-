#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import random
import copy
import itertools


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self, args, tao=0.2, gamma=0.7, beta=0.2, omega=0.5):
        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta),
                                          Pooling(cmp_ids=args.compress_id, omega=omega)]
        self.weights = [1] * len(self.data_augmentation_methods)
        print("Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, sequence):
        # randint generate int x in range: a <= x <= b
        aug_type = []
        augment_method_idx_1 = random.randint(0, len(self.data_augmentation_methods) - 1)
        aug_type.append(augment_method_idx_1)
        augment_method_1 = self.data_augmentation_methods[augment_method_idx_1]

        weights = self.weights.copy()
        weights[augment_method_idx_1] += len(weights) - 2

        augment_method_idx_2 = random.choices(list(range(len(weights))), weights, k=1)[0]
        aug_type.append(augment_method_idx_2)
        augment_method_2 = self.data_augmentation_methods[augment_method_idx_2]

        # print(augment_method.__class__.__name__) # debug usage
        return [augment_method_1(sequence), augment_method_2(sequence)], aug_type


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length < 1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index: start_index + sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, gamma=0.7):
        self.gamma = gamma

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma * len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copied_sequence[start_index: start_index + sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + copied_sequence[start_index + sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


class Pooling(object):
    """Randomly get a subsequence for pooling"""
    def __init__(self, cmp_ids, omega=0.5):
        self.omega = omega
        self.cmp_ids = cmp_ids

    def __call__(self, sequence):
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.omega * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length - 1)
        sub_seq = copy.deepcopy(copied_sequence[start_index: start_index + sub_seq_length])
        new_seq = copied_sequence[:start_index] + [self.cmp_ids] + copied_sequence[start_index + sub_seq_length:]
        return new_seq, sub_seq

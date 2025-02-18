# ------------------------------------------------------------------------------ #
# Author: Zhenwei Shao (https://github.com/ParadoxZW)
# Description: set const paths and dirs
# ------------------------------------------------------------------------------ #

import os


class PATH:
    def __init__(self):
        self.LOG_ROOT = 'outputs/logs/'
        self.CKPT_ROOT = 'outputs/ckpts/'
        self.RESULTS_ROOT = 'outputs/results/'
        self.DATASET_ROOT = 'datasets/'
        self.ASSETS_ROOT = 'assets/'

        self.IMAGE_DIR = {
            'train2014': self.DATASET_ROOT + 'coco2014/train2014/',
            'val2014': self.DATASET_ROOT + 'coco2014/val2014/',
            # 'test2015': self.DATASET_ROOT + 'coco2015/test2015/',
            'train2017': self.DATASET_ROOT + 'coco2017/train2017/',
            'val2017': self.DATASET_ROOT + 'coco2017/val2017/',
            'test2017': self.DATASET_ROOT + 'coco2017/test2017/',
            'unbalanced_train': self.DATASET_ROOT + 'unbalanced/train/',
            'unbalanced_test': self.DATASET_ROOT + 'unbalanced/test/',
            'balanced_10_train': self.DATASET_ROOT + 'balanced_10/train/',
            'balanced_10_test': self.DATASET_ROOT + 'balanced_10/test/',
        }

        self.FEATS_DIR = {
            'train2014': self.DATASET_ROOT + 'coco2014_feats/train2014/',
            'val2014': self.DATASET_ROOT + 'coco2014_feats/val2014/',
            'train2017': self.DATASET_ROOT + 'coco2017_feats/train2017/',
            'val2017': self.DATASET_ROOT + 'coco2017_feats/val2017/',
            'test2017': self.DATASET_ROOT + 'coco2017_feats/test2017/',
            'unbalanced_train': self.DATASET_ROOT + 'unbalanced_feats/train/',
            'unbalanced_test': self.DATASET_ROOT + 'unbalanced_feats/test/',
            'balanced_10_train': self.DATASET_ROOT + 'balanced_10_feats/train/',
            'balanced_10_test': self.DATASET_ROOT + 'balanced_10_feats/test/',
        }

        self.QUESTION_PATH = {
            'v2train': self.DATASET_ROOT + 'vqav2/v2_OpenEnded_mscoco_train2014_questions.json',
            'v2val': self.DATASET_ROOT + 'vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
            'vg': self.DATASET_ROOT + 'vqav2/vg_questions.json',
            'v2valvg_no_ok': self.DATASET_ROOT + 'vqav2/v2valvg_no_ok_questions.json',
            'oktrain': self.DATASET_ROOT + 'okvqa/OpenEnded_mscoco_train2014_questions.json',
            'oktest': self.DATASET_ROOT + 'okvqa/OpenEnded_mscoco_val2014_questions.json',
            'aoktrain': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_train.json',
            'aokval': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_val.json',
            'aoktest': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_test.json',
            'unbalanced_train': self.DATASET_ROOT + 'unbalanced/unbalanced_train_questions.json',
            'unbalanced_test': self.DATASET_ROOT + 'unbalanced/unbalanced_test_questions.json',
            'balanced_10_train': self.DATASET_ROOT + 'balanced_10/balanced_10_train_questions.json',
            'balanced_10_test': self.DATASET_ROOT + 'balanced_10/balanced_10_test_questions.json',
        }

        self.ANSWER_PATH = {
            'v2train': self.DATASET_ROOT + 'vqav2/v2_mscoco_train2014_annotations.json',
            'v2val': self.DATASET_ROOT + 'vqav2/v2_mscoco_val2014_annotations.json',
            'vg': self.DATASET_ROOT + 'vqav2/vg_annotations.json',
            'v2valvg_no_ok': self.DATASET_ROOT + 'vqav2/v2valvg_no_ok_annotations.json',
            'oktrain': self.DATASET_ROOT + 'okvqa/mscoco_train2014_annotations.json',
            'oktest': self.DATASET_ROOT + 'okvqa/mscoco_val2014_annotations.json',
            'aoktrain': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_train.json',
            'aokval': self.DATASET_ROOT + 'aokvqa/aokvqa_v1p0_val.json',
            'unbalanced_train': self.DATASET_ROOT + 'unbalanced/unbalanced_train_annotations.json',
            'unbalanced_test': self.DATASET_ROOT + 'unbalanced/unbalanced_test_annotations.json',
            'balanced_10_train': self.DATASET_ROOT + 'balanced_10/balanced_10_train_annotations.json',
            'balanced_10_test': self.DATASET_ROOT + 'balanced_10/balanced_10_test_annotations.json',
        }

        self.ANSWER_DICT_PATH = {
            'v2': self.ASSETS_ROOT + 'answer_dict_vqav2.json',
            'ok': self.ASSETS_ROOT + 'answer_dict_okvqa.json',
            'aok': self.ASSETS_ROOT + 'answer_dict_aokvqa.json',
            'unbalanced': self.ASSETS_ROOT + 'answer_dict_unbalanced.json',
            'balanced_10': self.ASSETS_ROOT + 'answer_dict_balanced_10.json',
        }

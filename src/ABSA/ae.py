# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
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


import logging
import numpy as np
import torch
import nltk

from torch.utils.data import TensorDataset, DataLoader
from ABSA import absa_data_utils as data_utils
from ABSA.absa_data_utils import ABSATokenizer

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('stopwords')
import string

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BERT():
    """ae model object"""
    def __init__(self, max_seq_length = 100, tokenizer_dir = "../model/pt_model/rest_pt", model_dir = "../model/ae/model.pt"):
        """
        :param max_seq_length: max length a sentence could be
        :param tokenizer_dir: pre-trained model
        :param model_dir: model directory
        """
        self.max_seq_length = max_seq_length
        self.tokenizer = ABSATokenizer.from_pretrained(tokenizer_dir)
        self.model = torch.load(model_dir)
        self.processor = data_utils.AeProcessor()
        self.label_list = self.processor.get_labels()
        self.stop = set(stopwords.words('english'))


    def text_tokenize(self,text):
        """tokenize the text"""
        sentences = []
        # break text into sentences
        sentences = sent_tokenize(text)
        # break sentence into words
        sentences = [word_tokenize(sentence) for sentence in sentences]
        return sentences

    def predict(self, text):
        """
        :param text: str - text in which aspects will be extracted
        :return: list[str] - list of aspects
        """
        sentences = self.text_tokenize(text)
        eval_examples = self.processor.get_input_examples(sentences)
        eval_features = data_utils.convert_examples_to_features(eval_examples, self.label_list, self.max_seq_length,
                                                                self.tokenizer, "ae")
        logger.info("***** Running preduction *****")
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

        eval_dataloader = DataLoader(eval_data)

        self.model.cuda()
        self.model.eval()

        full_logits = []
        full_label_ids = []
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, segment_ids, input_mask, label_ids = batch

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()

            full_logits.extend(logits.tolist())
            full_label_ids.extend(label_ids.tolist())

        assert len(full_logits) == len(eval_examples)
        # sort by original order for evaluation
        recs = {}
        for qx, ex in enumerate(eval_examples):
            recs[int(ex.guid.split("-")[1])] = {"sentence": ex.text_a, "idx_map": ex.idx_map,
                                                "logit": full_logits[qx][1:]}  # skip the [CLS] tag.
        full_logits = [recs[qx]["logit"] for qx in range(len(full_logits))]
        raw_X = [recs[qx]["sentence"] for qx in range(len(eval_examples))]
        # idx_map = [recs[qx]["idx_map"] for qx in range(len(eval_examples))]

        result = []
        for i in range(len(full_logits)):
            aspects = self.generate_word(raw_X[i],full_logits[i])
            # aspects = self.remove_sw_punct(aspects)
            if len(aspects) != 0:
                result = result + aspects
        return result

    def generate_word(self,sentence, logits):
        """
        :param sentence: text
        :param logits: probability value for each word
        :return: list[str] - a list of aspects
        """
        aspects = []
        word = ""
        for j in range(len(sentence)):
            max_word = self.label_list[np.argmax(logits[j])]
            if max_word == "I":
                if len(word) != 0:
                    word = word + " " + sentence[j]
            else:
                if len(word) != 0:
                    aspects.append([word.lower()])
                    word = ""
                if max_word == "B":
                    word = sentence[j]
        if len(word) != 0:
            aspects.append([word.lower()])
        return self.remove_sw_punct(aspects)

    def remove_sw_punct(self, aspects):
        """remove punctuation and stop words"""
        result = []
        for i,aspect in enumerate(aspects):
            temp_aspect = nltk.word_tokenize(aspect[0])
            temp_aspect = [t for t in temp_aspect if t not in self.stop and t not in string.punctuation]
            if len(temp_aspect) != 0:
                result = result + aspect
        return result







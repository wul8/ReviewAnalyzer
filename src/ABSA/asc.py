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

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
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

    """asc model object"""
    def __init__(self, max_seq_length=100, tokenizer_dir="../model/pt_model/rest_pt", model_dir="../model/asc/model.pt"):
        self.max_seq_length = max_seq_length
        self.tokenizer = ABSATokenizer.from_pretrained(tokenizer_dir)
        self.model = torch.load(model_dir)
        self.processor = data_utils.AscProcessor()
        self.label_list = self.processor.get_labels()


    def predict(self,text, aspects):  # Load a trained model that you have fine-tuned (we assume evaluate on cpu)
        """
        :param text: str - text of aspects
        :param aspects: list[str] - list of aspects
        :return: list[str] - list of sentiments
        """
        processor = data_utils.AscProcessor()
        label_list = processor.get_labels()
        eval_examples = processor.get_input_examples(text,aspects)
        eval_features = data_utils.convert_examples_to_features(eval_examples, label_list, self.max_seq_length, self.tokenizer,
                                                                "asc")

        logger.info("***** Running evaluation *****")
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)


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
        return [self.label_list[np.argmax(t)] for t in full_logits]


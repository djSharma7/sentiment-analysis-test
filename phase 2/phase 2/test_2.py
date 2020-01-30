from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook, trange


from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup as WarmupLinearSchedule
from utils import (convert_examples_to_features,
                        output_modes, processors, InputExample, convert_example_to_feature)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from torch import nn


def test_data(test_csv, tokenizer):
    if type(test_csv) == list:
        test_strings = test_csv[:]
    else:
        test_strings = pd.read_csv(test_csv, header=None)
        test_strings = test_strings[0].to_list()
    
    output_mode = args['output_mode']
    examples = [InputExample(guid=idx, text_a=test_strings[idx], text_b=None, label='Product') for idx in range(len(test_strings))]
    features = convert_examples_to_features(examples, ['Packing','Price','Product','Delivery','Sentiment'], args['max_seq_length'], tokenizer, output_mode,
            cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
            pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0,
            process_count=2)
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    print("length of final dataset: ", len(dataset))
    return dataset, test_strings



def test(test_csv, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    softmax = nn.Softmax(dim=1)
    eval_output_dir = args['output_dir']

    results = {}
    EVAL_TASK = args['task_name']
    print("eval task: ", EVAL_TASK)

    eval_dataset, test_strings = test_data(test_csv, tokenizer)
    
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    print("len eval data: ", len(eval_dataset))


    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0

    ans_preds = []
    ans_probs_neg = []
    ans_probs_pos = []

    preds = None
    probs = None
    out_label_ids = None
    for batch in tqdm_notebook(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids
#                       'labels':         batch[3]}
            outputs = model(**inputs)
            # print("outputs:----------------")
            # print(outputs)
            logits = outputs[0]
        
        # probs = softmax(logits)
        # probs = probs.detach().cpu().numpy()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            print (preds), " first"
            preds = np.argmax(preds, axis=1)

            probs = softmax(logits)
            probs = probs.detach().cpu().numpy()
#             out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            # print('probs shape inside else: ', probs.shape)
            print (preds), "predsss"
            preds = np.append(preds, np.argmax(logits.detach().cpu().numpy(), axis=1), axis=0)
            # print(probs)
            # print("----------------------------------")
            # print(softmax(logits).detach().cpu().numpy())
            probs = np.append(probs, softmax(logits).detach().cpu().numpy(), axis=0)
    print (test_strings)
    print (preds)
    results = pd.DataFrame(list(zip(test_strings, preds, probs[:, 0], probs[:, 1], probs[:,2], probs[:,3], probs[:,4], probs[:,5])),
               columns =['string', 'prediction', 'Delivery', 'Packaging', 'Price', 'Product','negative','positive'])
    
    results.to_csv('test_results.csv', index=None)
    
    return results, probs



with open('args.json', 'r') as fp:
    args = json.load(fp)


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]



config = config_class.from_pretrained(args['model_name'], num_labels=6, finetuning_task=args['task_name'])
tokenizer = tokenizer_class.from_pretrained(args['model_name'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task = args['task_name']

processor = processors[task]()
label_list = processor.get_labels()
num_labels = len(label_list)
print (num_labels, "labels list")


model_dir = 'outputs/'
model = model_class.from_pretrained(model_dir)
model.to(device);


# test_strings = ['Product is good', 'sahi time pe mil gaya', 'compared to price, expected quality of product is good', 'except delivery, price is good']
Input_file = 'test_002.csv'
if __name__ == '__main__':
    preds, probs = test(Input_file, model, tokenizer)


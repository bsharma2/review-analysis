from bertviz_repo.bertviz.transformers_neuron_view import BertModel, BertTokenizer
import torch
from torch.utils.data import (DataLoader, TensorDataset)
from utils_glue import InputExample,_truncate_seq_pair,InputFeatures
from pytorch_transformers import BertTokenizer
from collections import defaultdict

# output_dir = "./my_BERT"
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed
# PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
# model_loaded = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
# checkpoint = torch.load(output_dir + "/sentiment_analysis_model.pt")

# checkpoint_model_state_dict = checkpoint['model_state_dict']
# for key in list(checkpoint_model_state_dict.keys()):
#     if key.startswith('bert.'):
#         new_key = key.replace("bert.", "")
#         checkpoint_model_state_dict.update({new_key: checkpoint_model_state_dict[key]})
#         checkpoint_model_state_dict.pop(key)
#     if key.startswith('cls_layer.'):
#         checkpoint_model_state_dict.pop(key)
#     if key.startswith('out.'):
#         checkpoint_model_state_dict.pop(key)

# model_loaded.load_state_dict(checkpoint['model_state_dict'])

# model = model_loaded.to(device)
# model.eval()

  
def convert_to_features(examples,tokenizer,cls_token_at_end=False,max_seq_length = 128, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids,
                                label_id=-1))
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    eval_dataloader = DataLoader(dataset,batch_size=8)
    for batch in eval_dataloader:
        batch = tuple(t.to("cpu") for t in batch)
        inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1]
                        # 'token_type_ids': batch[2],  # XLM don't use segment_id
              }
    return inputs,[cls_token]+tokens_a+[sep_token],tokens_b+[sep_token]

# infer_model,infer_tokenizer, infer_config = load_artifacts()
# model, tokenizer = load_artifacts()



#---- Visualization

def _decode_output(attentions, tokens_a, tokens_b):
  attn_dict = defaultdict(list)
  slice_a = slice(0, len(tokens_a))  # Positions corresponding to sentence A in input
  slice_b = slice(len(tokens_a), len(tokens_a) + len(tokens_b))  # Position corresponding to sentence B in input

  for layer,attn_data in enumerate(attentions):
    attn = attn_data['attn'][0]
    attn_dict['all'].append(attn.tolist())
    attn_dict['aa'].append(attn[:, slice_a, slice_a].tolist())  # Append A->A attention for layer, across all heads
    attn_dict['bb'].append(attn[:, slice_b, slice_b].tolist())  # Append B->B attention for layer, across all heads
    attn_dict['ab'].append(attn[:, slice_a, slice_b].tolist())  # Append A->B attention for layer, across all heads
    attn_dict['ba'].append(attn[:, slice_b, slice_a].tolist())  # Append B->A attention for layer, across all heads
  results = {
    'all': {
        'attn': attn_dict['all'],
        'left_text': tokens_a + (tokens_b if tokens_b else []),
        'right_text': tokens_a + (tokens_b if tokens_b else [])
    }
  }

  results.update({
        'aa': {
            'attn': attn_dict['aa'],
            'left_text': tokens_a,
            'right_text': tokens_a
        },
        'bb': {
            'attn': attn_dict['bb'],
            'left_text': tokens_b,
            'right_text': tokens_b
        },
        'ab': {
            'attn': attn_dict['ab'],
            'left_text': tokens_a,
            'right_text': tokens_b
        },
        'ba': {
            'attn': attn_dict['ba'],
            'left_text': tokens_b,
            'right_text': tokens_a
        }
    });

  return results


def disp_attn_tokens(tokens_a,viz_attn_dict,layer,head,attn_source_word_index=-1,attn_direction='ba'):
  max_alpha = 0.1
  weights = []
  words = []

  for word,weight in zip(tokens_a,viz_attn_dict[attn_direction]['attn'][layer][head][attn_source_word_index]):
      weights.append(weight)
      words.append(str(word))
      # if weight is not None:
      #     print("{0}:{1}".format(word,weight))

  return weights,words



aspect =  "what do you think of the product" #@param ["what do you think of the product"]{type:"string"}
layer = 10 
head = 4

def result(review_text):

  output_dir = "./my_BERT"
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed
  PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
  model_loaded = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

  checkpoint = torch.load(output_dir + "/sentiment_analysis_model.pt")

  checkpoint_model_state_dict = checkpoint['model_state_dict']
  for key in list(checkpoint_model_state_dict.keys()):
      if key.startswith('bert.'):
          new_key = key.replace("bert.", "")
          checkpoint_model_state_dict.update({new_key: checkpoint_model_state_dict[key]})
          checkpoint_model_state_dict.pop(key)
      if key.startswith('cls_layer.'):
          checkpoint_model_state_dict.pop(key)
      if key.startswith('out.'):
          checkpoint_model_state_dict.pop(key)

  model_loaded.load_state_dict(checkpoint['model_state_dict'])

  model = model_loaded.to(device)
  model.eval()

  viz_examples = [InputExample(guid=0, text_a=review_text, text_b=aspect, label=None)]
  viz_inputs,tokens_a,tokens_b = convert_to_features(viz_examples,tokenizer)
  viz_ouputs = model(**viz_inputs)
  viz_attn_dict = _decode_output(viz_ouputs[-1], tokens_a, tokens_b)
  weights, words = disp_attn_tokens(tokens_a, viz_attn_dict, layer=layer, head=head)
  return weights, words

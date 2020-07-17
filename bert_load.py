from transformers import BertModel, BertTokenizer
import torch

def method_load(review_text):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	output_dir = "./my_BERT"
	tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed
	review_text = [review_text]
	MAX_LEN= 160
	for text in review_text:
		encoded_review = tokenizer.encode_plus(
			text,
			max_length=MAX_LEN,
			add_special_tokens=True,
			return_token_type_ids=True,
			pad_to_max_length=True,
			return_attention_mask=True,
			return_tensors='pt',
		)
	input_ids = encoded_review['input_ids'].to(device)
	attention_mask = encoded_review['attention_mask'].to(device)


	return input_ids, attention_mask
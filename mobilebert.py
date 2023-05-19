import torch
from transformers import AutoTokenizer, AutoConfig, MobileBertModel
import numpy as np


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    array = token_embeddings.numpy()
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# load model
model_name = r'cycloneboy/chinese_mobilebert_base_f2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = MobileBertModel.from_pretrained(model_name, config=config)

sentences = ['如何更换花呗绑定银行卡']
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
print(encoded_input)

# dummy_input = torch.ones(1, 64).int()
# input_ids = torch.ones(1, 64, 768).float()
# token_type_ids = torch.ones(1, 64).int()
# attention_mask = torch.ones(1, 64).int()
#
# input_names = ['input_ids', 'attention_mask', 'token_type_ids']
# onnx_path = r".\model\mobilebert_model_64_cut_ful.onnx"
# torch.onnx.export(model, (input_ids, token_type_ids, attention_mask), onnx_path,
#                   opset_version=11, input_names=input_names)


with torch.no_grad():
    model_output = model(**encoded_input)
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print("Sentence embeddings: ", sentence_embeddings.size())
np.savetxt("embeddings.txt", sentence_embeddings)

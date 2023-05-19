from transformers import BertTokenizer, BertModel
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')
'''
sentences = ['如何更换花呗绑定银行卡']
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
print(encoded_input)
# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

print(model_output)
# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print("Sentence embeddings:")
print(sentence_embeddings)
'''

# 导出PyTorch模型为ONNX
# dummy_input = torch.ones(1, 13).int()
# dummy_input = torch.ones(1, 64).int()
dummy_input = torch.ones(1, 64, 768).float()
# onnx_path = r".\model\text2vec_model_13.onnx"
# onnx_path = r".\model\text2vec_model_64.onnx"
onnx_path = r".\model\text2vec_model_64_cut.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)

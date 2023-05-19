import onnxruntime as ort
import numpy as np
import onnx
import copy
from collections import OrderedDict
import torch

# model_path = r'.\model\text2vec_model_13_op11.onnx'
# model = onnx.load(model_path)
# # 模型推理
# ori_output = copy.deepcopy(model.graph.output)
# # 输出模型每层的输出
# for node in model.graph.node:
#     for output in node.output:
#         model.graph.output.extend([onnx.ValueInfoProto(name=output)])
#
# session = ort.InferenceSession(model.SerializeToString())
#
# array = np.array([[101, 1963, 862, 3291, 2940, 5709, 1446, 5308, 2137, 7213, 6121, 1305, 102]])
# ort_inputs = {'input.1': array}
# ort_outs = session.run(None, ort_inputs)
# # 获取所有节点输出
# outputs = [x.name for x in session.get_outputs()]
# # 生成字典，便于查找层对应输出
# ort_outs = OrderedDict(zip(outputs, ort_outs))
# # print(ort_outs['onnx::Add_221'])
#
# # 将有序字典存储到文本文件中
# file_path = r'.\result\text2vec_dict_13.txt'
# with open(file_path, 'w') as file:
#     for key, value in ort_outs.items():
#         file.write(f"{key}: {value}\n")


# model_path = r'.\model\text2vec_model_64.onnx'
# model = onnx.load(model_path)
# # 模型推理
# ori_output = copy.deepcopy(model.graph.output)
# # 输出模型每层的输出
# for node in model.graph.node:
#     for output in node.output:
#         model.graph.output.extend([onnx.ValueInfoProto(name=output)])
#
# session = ort.InferenceSession(model.SerializeToString())
#
# input_ids = np.zeros((1, 64), dtype=np.int32)
# values = [101, 1963, 862, 3291, 2940, 5709, 1446, 5308, 2137, 7213, 6121, 1305, 102]
# input_ids[:, :13] = values
#
# ort_inputs = {'input.1': input_ids}
# ort_outs = session.run(None, ort_inputs)
# # 获取所有节点输出
# outputs = [x.name for x in session.get_outputs()]
# # 生成字典，便于查找层对应输出
# ort_outs = OrderedDict(zip(outputs, ort_outs))
# Add_221 = ort_outs['onnx::Add_221']
# print("Add_221: ", Add_221)
#
# # 将有序字典存储到文本文件中
# file_path = r'.\result\text2vec_dict_64.txt'
# with open(file_path, 'w') as file:
#     for key, value in ort_outs.items():
#         file.write(f"{key}: {value}\n")
#
# model_path_1 = r'.\model\text2vec_model_64_cut.onnx'
# session_1 = ort.InferenceSession(model_path_1)
# output = session_1.run(None, {'onnx::Add_0': Add_221})
# print("Cut output: ", output)
# np.savetxt(r".\result\Gather_1423.txt", output[0].squeeze())
#
# # 构造SNPE输入
# input_1 = np.transpose(Add_221, (0, 2, 1))
# input_1.tofile(r".\data\Add_221.raw")
# np.savetxt(r".\data\Add_221.txt", input_1.squeeze())


# ''' mobilebert mean pooling
model_path = r'.\model\mobilebert_model_64_ful.onnx'
model = onnx.load(model_path)
# 模型推理
ori_output = copy.deepcopy(model.graph.output)
# 输出模型每层的输出
for node in model.graph.node:
    for output in node.output:
        model.graph.output.extend([onnx.ValueInfoProto(name=output)])

session = ort.InferenceSession(model.SerializeToString())

input_ids = np.zeros((1, 64), dtype=np.int64)
values = [101, 1963, 862, 3291, 2940, 5709, 1446, 5308, 2137, 7213, 6121, 1305, 102]
input_ids[:, :13] = values
token_type_ids = np.zeros((1, 64), dtype=np.int64)
attention_mask = np.zeros((1, 64), dtype=np.int64)
attention_mask[:, :13] = 1

ort_inputs = {'input.1': input_ids,
              'input.5': token_type_ids,
              'attention_mask': attention_mask}
ort_outs = session.run(None, ort_inputs)
# 获取所有节点输出
outputs = [x.name for x in session.get_outputs()]
# 生成字典，便于查找层对应输出
ort_outs = OrderedDict(zip(outputs, ort_outs))
output = ort_outs['onnx::Gather_1627']


def mean_pooling(output, attention_mask):
    token_embeddings = torch.from_numpy(output)
    attention_mask = torch.from_numpy(attention_mask)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


sentence_embeddings = mean_pooling(output, attention_mask)
print("Sentence embeddings: ", sentence_embeddings)

# '''
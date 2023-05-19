# Chinese MobileBert with SNPE

## 简介

如何在Qualcomm SnapDragon 865 LU上跑Chinese MobileBert模型。

开源模型：

- [shibing624/text2vec-base-chinese](https://huggingface.co/shibing624/text2vec-base-chinese/tree/main)

- [cycloneboy/chinese_mobilebert_base_f2](https://huggingface.co/cycloneboy/chinese_mobilebert_base_f2)

参考资料：

- [HuggingFace快速上手（以bert-base-chinese为例）](https://zhuanlan.zhihu.com/p/610171544)

- [c++ version of bert tokenize](https://gist.github.com/luistung/4f23b7d0026b26560fdd82a3b39ca460)

## 导出onnx

```python
# torch模型导出onnx比较简单
dummy_input = torch.ones(1, 13).int()
# dummy_input = torch.ones(1, 64).int()
# dummy_input = torch.ones(1, 64, 768).float()
onnx_path = r".\text2vec_model_13.onnx"
# onnx_path = r".\text2vec_model_64.onnx"
# onnx_path = r".\text2vec_model_64_cut.onnx"
torch.onnx.export(model, dummy_input, onnx_path, opset_version=11)
```

- 需要关注模型输入，所以可以先打印一下encoded_input，可以看到模型输入由input_ids，token_type_ids，attention_mask三部分组成，具体含义可以看文档中的tokenizer章节介绍 。
- 使用句子**“如何更换花呗绑定银行卡”**作为样例输入，导出onnx以1x13（CLS+Sentence+SEP）的input_ids尺寸为例（为什么这里先导出单输入模型，是因为snpe-net-run的限制，时间上不允许深入研究snpe-net-run多输入的方法），这个模型的token_type_ids的定义和文档里的不太一样，为全0，所以可以直接使用默认值；另外attention_mask，因为我们只考虑单输入，那么将默认为全1，会导致onnx的推理结果和pt的不一致，这是需要注意的。
- 目前onnx最新版本为13，使用opset=11是因为snpe对11版本的支持更好。

## 模型转换及验证

- 模型转换：v1.x转换失败，v2.x转换成功，根据高通官方给的信息，用snpe-2.7以上版本会有更好的兼容性

```bash
snpe-onnx-to-dlc --input_network text2vec_13.onnx
                 --output_path text2vec_13_v2.7.dlc
```

- 输入/输出：在SNPE的定义下不管是输入还是输出都是numpy array，可以使用np.tofile()和np.fromfile()进行操作

```python
# create_raw.py
import numpy as np

data = [101, 1963, 862, 3291, 2940, 5709, 1446, 5308, 2137, 7213, 6121, 1305, 102]
array = np.array(data)
# 保存数组到文件中
array.tofile('test_13.raw')

# snpe-net-run --container text2vec_13_v2.7.dlc --input_list input.txt

# read_raw.py
import numpy as np

float_array = np.fromfile(r"result_13_v2.7.raw", dtype=np.float32)
array = float_array.reshape((1, 13, 768))
np.savetxt("result_13_v2.7.txt", array.squeeze())
```

- 输出验证，结果不一致时我的一些调查思路
    - 输入：vim十六进制模式可以看输入到底对不对，实际调研时第一个考虑的方向是会不会是数据大小端导致的，于是强制修改了输入为一个极大值，在跑onnx和pt时均在Gather节点报错，提示输入最大为21128，查了下网络结构，可以看出来Gather节点的21128个权重向量对应了单词表里的21128个单词，但是在snpe-net-run却能正常运行，这是不对的，因此我换了多个输入，发现不同输入的输出完全一样。
    - 二分查找第一个输出不一样的层：将onnx的每一层输出保存下来，snpe-net-run指定输出层
    - 由于有了**”不同输入的输出完全一样“**的线索，所以我直接将问题定位至Gather节点的输出层，这时将snpe-net-run的--set_output_tensors设置为onnx::Add_221，并进行对比，发现snpe的输出为64个相同的向量，并且为Gather节点的第一个向量，这就意味着不管我输入什么，在Gather节点都被当作全0处理了。

## 网络剪枝

- forward的修改，去掉Gather节点，重新导onnx，因为需求输入为64的句子，所以这时候改输入为1x64，由于去掉了Gather节点，所以输入变为1x64x768，在封装sdk时需要自行实现Gather运算。

```python
# modeling_bert.py
# Line 209
				if input_ids is not None:
            # input_shape = input_ids.size()
            input_shape = input_ids.size()[:-1]
# Line 231
				if inputs_embeds is None:
				    # inputs_embeds = self.word_embeddings(input_ids)
				    inputs_embeds = input_ids
# Line 961
        elif input_ids is not None:
            # input_shape = input_ids.size()
            input_shape = input_ids.size()[:-1]
```

- 剪枝后的输入如何获取：转成dlc之后输入有一个transpose，从1x64x768变成1x768x64了，所以input_1也对应的做了个transpose

```python
import onnxruntime as ort
import numpy as np
import onnx
import copy
from collections import OrderedDict

model_path = r'.\model\text2vec_model_64.onnx'
model = onnx.load(model_path)
# 模型推理
ori_output = copy.deepcopy(model.graph.output)
# 输出模型每层的输出
for node in model.graph.node:
    for output in node.output:
        model.graph.output.extend([onnx.ValueInfoProto(name=output)])

session = ort.InferenceSession(model.SerializeToString())

input_ids = np.zeros((1, 64), dtype=np.int32)
values = [101, 1963, 862, 3291, 2940, 5709, 1446, 5308, 2137, 7213, 6121, 1305, 102]
input_ids[:, :13] = values

ort_inputs = {'input.1': input_ids}
ort_outs = session.run(None, ort_inputs)
# 获取所有节点输出
outputs = [x.name for x in session.get_outputs()]
# 生成字典，便于查找层对应输出
ort_outs = OrderedDict(zip(outputs, ort_outs))
Add_221 = ort_outs['onnx::Add_221']
print("Add_221: ", Add_221)

# 将有序字典存储到文本文件中
file_path = r'.\result\text2vec_dict_64.txt'
with open(file_path, 'w') as file:
    for key, value in ort_outs.items():
        file.write(f"{key}: {value}\n")

model_path_1 = r'.\model\text2vec_model_64_cut.onnx'
session_1 = ort.InferenceSession(model_path_1)
output = session_1.run(None, {'onnx::Add_0': Add_221})
print("Cut output: ", output)
np.savetxt("Gather_1423.txt", output[0].squeeze())

# 构造SNPE输入
input_1 = np.transpose(Add_221, (0, 2, 1))
input_1.tofile("Add_221.raw")
np.savetxt("Add_221.txt", input_1.squeeze())
```

## 多输入模型

- mobilebert和bert基础网络结构一致，去除Gather节点的方法一样
- 导onnx时设置输入尺寸为tuple类型，并指定输入名（因为输入维度一样，默认情况不一定对的上，所以需要调整）

```python
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
# model.eval()

sentences = ['如何更换花呗绑定银行卡']
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
print(encoded_input)

dummy_input = torch.ones(1, 64).int()
input_ids = torch.ones(1, 64, 768).float()
token_type_ids = torch.ones(1, 64).int()
attention_mask = torch.ones(1, 64).int()

input_names = ['input_ids', 'attention_mask', 'token_type_ids']
onnx_path = r".\model\mobilebert_model_64_cut_ful.onnx"
torch.onnx.export(model, (input_ids, token_type_ids, attention_mask), onnx_path,
                  opset_version=11, input_names=input_names)

with torch.no_grad():
    model_output = model(**encoded_input)
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print("Sentence embeddings: ", sentence_embeddings.size())
np.savetxt("embeddings.txt", sentence_embeddings)
```

## SDK封装

### c++ version of bert tokenize

- vocab.txt附在模型库中
- 需要手动插入CLS和SEP

### Gather & mean pooling

```cpp
// Gather weights保存成numpy array bin的格式
void MobileBert::ReadWeights(std::vector<std::vector<float>>& weights, const std::string& path) {
    const size_t rows = weights.size();
    const size_t cols = weights[0].size();
    std::vector<float> data(rows * cols);

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Weights file: {} open failed.", path);
        return ;
    }
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    file.close();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            weights[i][j] = data[i * cols + j];
        }
    }
}

void MobileBert::Preprocess(const std::string& text) {
    int size = text.size();
    auto tokens = m_tokenizer->tokenize(text);
    auto ids = m_tokenizer->convertTokensToIds(tokens);
    ids.insert(ids.begin(), VOCAB_BEGIN);
    ids.emplace_back(VOCAB_END);
    auto input_ids = m_task->getInputTensor(m_inputLayers[0]);
    auto attenrtion_mask = m_task->getInputTensor(m_inputLayers[1]);
    memset(input_ids, 0, sizeof(input_ids));
    memset(attenrtion_mask, 0, sizeof(attenrtion_mask));

    // Gather: get indices & transpose
    for (int i = 0; i < 768; ++i) {
        for (int j = 0; j < ids.size(); ++j) {
            input_ids[i * 64 + j] = m_inputWeights[ids[j]][i];
            attenrtion_mask[j] = 1;
        }
    }
    for (int i = 0; i < 768; ++i) {
        for (int j = ids.size(); j < 64; ++j) {
            input_ids[i * 64 + j] = m_inputWeights[0][i];
            attenrtion_mask[j] = 0;
        }
    }
    m_inputIdsLen = ids.size();
}

void MobileBert::Postprocess(BertOutput& result, int64_t time) {
    auto output = m_task->getOutputTensor(m_outputTensors[0]);
    auto outputShape = m_task->getOutputShape(m_outputTensors[0]);
    result.outputs.resize(outputShape[1], std::vector<float>(outputShape[2]));
    result.meanPoolingResults.resize(outputShape[2], 0.0f);
    // mean pooling
    for (int i = 0; i < outputShape[1]; ++i) {          // 64
        for (int j = 0; j < outputShape[2]; ++j) {      // 768
            result.outputs[i][j] = output[i * outputShape[2] + j];
            if (i < m_inputIdsLen) result.meanPoolingResults[j] += output[i * outputShape[2] + j];
        }
    }

    for (int j = 0; j < outputShape[2]; ++j) {      // 768
        result.meanPoolingResults[j] /= m_inputIdsLen;
    }
    
    result.time = time;
}
```

### SNPETask

见之前的视频：[基于高通SNPE推理引擎的yolov5目标检测算法——源码分析（1）](https://www.bilibili.com/video/BV18X4y197wd/?spm_id_from=333.999.0.0&vd_source=dbab91d49299bee362c0be8725a8da70)

### 编译运行
```shell
cd snpe_bert
mkdir build & cd build
cmake ..
make
./test-text2vec
```
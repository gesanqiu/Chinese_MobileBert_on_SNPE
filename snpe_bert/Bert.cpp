#include "Bert.h"

namespace bert {

MobileBert::MobileBert() : m_task(nullptr) {

}

MobileBert::~MobileBert() {
    DeInit();
}

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

bool MobileBert::Init(const BertConfig& config) {
    m_task = std::shared_ptr<snpetask::SNPETask>(new snpetask::SNPETask());
    m_tokenizer = std::shared_ptr<FullTokenizer>(new FullTokenizer(config.vocabPath));

    m_inputLayers = config.inputLayers;
    m_outputLayers = config.outputLayers;
    m_outputTensors = config.outputTensors;

    m_task->setOutputLayers(m_outputLayers);
    if (!m_task->init(config.modelPath, config.runtime)) {
        LOG_ERROR("Can't init snpetask instance.");
        return false;
    }

    m_inputWeights.resize(21128, std::vector<float>(768, 0));
    ReadWeights(m_inputWeights, config.inputWeightsPath);

    // token_type_ids不考虑
    // m_tokenWeights.resize(2, std::vector<int>(768, 0));
    // ReadWeights(m_tokenWeights, config.tokenWeightsPath);
    return true;
}

bool MobileBert::DeInit() {
    return true;
}

bool MobileBert::Inference(const std::string& text, BertOutput& result) {
    Preprocess(text);
    int64_t start = GetTimeStamp_ms();
    if (!m_task->execute()) {
        LOG_ERROR("SNPETask execute failed.");
        return false;
    }
    Postprocess(result, GetTimeStamp_ms() - start);
    return true;
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

}
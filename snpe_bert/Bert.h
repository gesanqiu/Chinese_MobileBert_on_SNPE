#pragma once

#include <vector>
#include <cstring>
#include <memory>

#include "SNPETask.h"
#include "tokenization.h"

namespace bert {

const size_t VOCAB_BEGIN = 101;
const size_t VOCAB_END   = 102;

struct BertConfig {
    std::string modelPath;
    std::string vocabPath;
    std::string inputWeightsPath;
    std::string tokenWeightsPath;
    runtime_t runtime;
    std::vector<std::string> inputLayers;
    std::vector<std::string> outputLayers;
    std::vector<std::string> outputTensors;
    int max_length = 64;
};

struct BertOutput {
    std::vector<std::vector<float>> outputs;    // [1x64x768]
    std::vector<float> meanPoolingResults;      // [1x768]
    int64_t time;                               // Inference time cost.
};

class MobileBert {
public:
    MobileBert();
    ~MobileBert();

    bool Init(const BertConfig& config);
    bool DeInit();
    bool Inference(const std::string& text, BertOutput& result);

private:
    void Preprocess(const std::string& text);
    void Postprocess(BertOutput& result, int64_t time);
    void ReadWeights(std::vector<std::vector<float>>& weights, const std::string& path);

    std::shared_ptr<snpetask::SNPETask> m_task;
    std::vector<std::string> m_inputLayers;
    std::vector<std::string> m_outputLayers;
    std::vector<std::string> m_outputTensors;
    std::shared_ptr<FullTokenizer> m_tokenizer;
    int m_maxLength = 64;
    int m_features = 768;
    int m_inputIdsLen = 64;

    std::vector<std::vector<float>> m_inputWeights;
    std::vector<std::vector<float>> m_tokenWeights;
};

}
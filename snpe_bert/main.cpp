#include <jsoncpp/json/json.h>

#include "Bert.h"

static runtime_t device2runtime(std::string&& device)
{
    std::transform(device.begin(), device.end(), device.begin(),
        [](unsigned char ch){ return tolower(ch); });

    if (0 == device.compare("cpu")) {
        return CPU;
    } else if (0 == device.compare("gpu")) {
        return GPU;
    } else if (0 == device.compare("gpu_float16")) {
        return GPU_FLOAT16;
    } else if (0 == device.compare("dsp")) {
        return DSP;
    } else if (0 == device.compare("aip")) {
        return AIP;
    } else { 
        return CPU;
    }
}

void ParseConfig(std::string path, bert::BertConfig& config)
{
    Json::Reader reader;
    Json::Value root;
    std::ifstream in(path, std::ios::binary);
    reader.parse(in, root);
    config.modelPath = root["model-path"].asString();
    config.runtime = device2runtime(root["runtime"].asString());
    config.vocabPath = root["vocab-path"].asString();
    config.inputWeightsPath = root["input-weights-path"].asString();
    if (root["input-layers"].isArray()) {
        int sz = root["input-layers"].size();
        for (int i = 0; i < sz; ++i)
            config.inputLayers.push_back(root["input-layers"][i].asString());
    }
    if (root["output-layers"].isArray()) {
        int sz = root["output-layers"].size();
        for (int i = 0; i < sz; ++i)
            config.outputLayers.push_back(root["output-layers"][i].asString());
    }
    if (root["output-tensors"].isArray()) {
        int sz = root["output-tensors"].size();
        for (int i = 0; i < sz; ++i)
            config.outputTensors.push_back(root["output-tensors"][i].asString());
    }
}

int main() {
    bert::BertConfig config;
    ParseConfig("../config.json", config);
    std::shared_ptr<bert::MobileBert> instance = std::shared_ptr<bert::MobileBert>(new bert::MobileBert());;
    std::string text = "如何更换花呗绑定银行卡";
    bert::BertOutput output;
    instance->Init(config);
    instance->Inference(text, output);
    LOG_INFO("Time cost: {}", output.time);
    return 0;
}

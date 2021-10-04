// {% copyright %}

#pragma once

#include "Op.h"

#include <layers/param_custom_cpp.h>
#include <common_types.h>

struct CustomCppLayerParams {
    uint32_t leonPreambleID;

    const uint8_t* kernelData;
    size_t kernelDataLen;

    uint32_t* paramData;
    sw_params::BaseKernelParams baseParamData;
    size_t paramDataLen;
    uint64_t kernel = 0;
};

class CustomCpp : public Op
{
public:
    CustomCpp() : Op(kCustomCpp) {};
    CustomCpp(t_MvTensorOpType /*op_type*/) : Op(kCustomCpp) {};
    virtual ~CustomCpp() override;

    virtual void run(mv::tensor::Processor& mvtp,
            t_MvTensorMyriadResources& myriadRes,
            t_MvTensorDebugInfo& debugInfo) override;
    void addInputBuffer(const Buffer& input, sw_params::Location loc = sw_params::Location::DDR) {
        inputVec.push_back(input);
        inputLocations.push_back(loc);
    }
    void addOutputBuffer(const Buffer& output, sw_params::Location loc = sw_params::Location::DDR) {
        outputVec.push_back(output);
        outputLocations.push_back(loc);
    }

    virtual bool parse(Layer *layer) override;

    CustomCppLayerParams ops;

private:
    std::vector<Buffer> inputVec;
    std::vector<sw_params::Location> inputLocations;
    std::vector<Buffer> outputVec;
    std::vector<sw_params::Location> outputLocations;
};

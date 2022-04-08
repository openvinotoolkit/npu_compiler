//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

namespace op_custom_ocl
{
    static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
        const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
        [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
        std::string&) -> std::pair<bool, std::size_t>
    {
        return {true, 0};
    };

    static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
        std::vector<Tensor>&)> outputDefFcn =
        [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
           std::vector<Tensor>& outputs)
    {
        // dType argument is ignored (output data type is stored in tensorInfo)
        const auto outputsInfo = args.at("outputsInfo").get<std::vector<mv::TensorInfo>>();

        for (size_t i = 0; i < outputsInfo.size(); i++) {
            auto dType = outputsInfo[i].type();
            if (dType == mv::DType("Default")) {
                dType = inputs[0]->getDType();
            }

            outputs.emplace_back(":" + std::to_string(i), outputsInfo[i].shape(), dType,
                                 outputsInfo[i].order());
        }
    };

}

namespace op {

    MV_REGISTER_OP(CustomOcl)
            .setInputs({"inputs"})
            .setOutputs({"outputs"})
            .setVariableInputNum(true)
            .setArg<std::vector<uint8_t>>("kernelData")
            .setArg<std::vector<uint8_t>>("paramData")
            .setArg<std::vector<mv::TensorInfo>>("outputsInfo")
            .setInputCheck(op_custom_ocl::inputCheckFcn)
            .setOutputDef(op_custom_ocl::outputDefFcn)
            .setTypeTrait({"executable", "exposed"});

}

}

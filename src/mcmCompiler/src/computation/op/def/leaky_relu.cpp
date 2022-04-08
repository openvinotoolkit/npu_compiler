//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_leaky_relu
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            if (inputs[0]->getShape().ndims() != 4)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 4, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());

                return {false, 0};
            }

            auto alpha = args.at("alpha").get<double>();
            
            // prelu may be implemented as leaky_relu, so alpha may be negative
            // -1 is a experimental value, we don't expect too large slope even for prelu
            if (alpha < -1)
            {
                errMsg = "Invalid slope (must be larger than -1): alpha=" + std::to_string(alpha);

                return {false, 0};
            }

            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& /*args*/, std::vector<Tensor>& outputs)
        {
            outputs.emplace_back(":0", inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder());
        };

    }

    namespace op {
        MV_REGISTER_OP(LeakyRelu)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setOptionalArg<double>("alpha", 0)
        .setInputCheck(op_leaky_relu::inputCheckFcn)
        .setOutputDef(op_leaky_relu::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}

//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_exp
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& /*args*/,
            std::string& /*errMsg*/) -> std::pair<bool, std::size_t>
        {
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

        MV_REGISTER_OP(Exp)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setInputCheck(op_exp::inputCheckFcn)
        .setOutputDef(op_exp::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}

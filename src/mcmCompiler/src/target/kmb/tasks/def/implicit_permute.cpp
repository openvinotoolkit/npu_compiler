//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_implicit_permute
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
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto outputOrder = inputs[0]->getOrder();
            auto outputShape = args.at("shape").get<mv::Shape>();

            outputs.emplace_back(":0", outputShape, inputs[0]->getDType(), outputOrder);
        };

        static std::string empty;

    }

    namespace op {

        MV_REGISTER_OP(ImplicitPermute)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<mv::Shape>("shape")
        .setInputCheck(op_implicit_permute::inputCheckFcn)
        .setOutputDef(op_implicit_permute::outputDefFcn);

    }
}

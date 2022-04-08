//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{
    namespace op_tile
    {
        // Only 4D tensors is supported right now, so axis value should be in range [0,3]
        #define MAX_AXIS_VALUE 3

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto input = inputs[0];
            auto inputShape = input->getShape();

            auto axis = MAX_AXIS_VALUE - args.at("axis").get<unsigned>();
            auto tiles = args.at("tiles").get<unsigned>();

            if (axis >= inputShape.ndims()) {
                errMsg = "Invalid axis number - has to be less than number of dimensions - 1"
                    + std::to_string(axis);
                return {false, 1};
            }

            if (tiles == 0) {
                errMsg = "Number of tiles shall be more than 0";
                return {false, 2};
            }

            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            mv::Order order(inputs[0]->getOrder());

            auto axis = MAX_AXIS_VALUE - args.at("axis").get<unsigned>();
            auto tiles = args.at("tiles").get<unsigned>();

            // calculate output shape from input shape and params
            auto inputShape = inputs[0]->getShape();
            auto ndims = inputShape.ndims();
            mv::Shape outputShape(ndims);
            for (size_t i = 0; i < ndims; i++)
            {
                outputShape[i] = inputShape[i];
                if (i == axis) {
                    outputShape[i] = outputShape[i] * tiles;
                }
            }

            outputs.emplace_back(":0",  outputShape, inputs[0]->getDType(), order);
        };
    }

    namespace op {

        // Tile layer extends input blob with copies of data along specific axis.

        MV_REGISTER_OP(Tile)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<unsigned>("axis")
        .setArg<unsigned>("tiles")
        .setInputCheck(op_tile::inputCheckFcn)
        .setOutputDef(op_tile::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}

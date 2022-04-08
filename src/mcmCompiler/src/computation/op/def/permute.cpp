//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_permute
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            auto input = inputs[0];
            if (inputs.size() != 1)
            {
                std::stringstream err;
                err << "Too many inputs (must be 1): " << inputs.size();
                errMsg = err.str();
                return {false, 0};
            }

            // New order must be a permutation of old order:
            auto old_order = input->getOrder();
            auto new_order = args.at("order").get<mv::Order>();
            auto old_order_str = old_order.toString();
            auto new_order_str = new_order.toString();
            std::sort(old_order_str.begin(), old_order_str.end());
            std::sort(new_order_str.begin(), new_order_str.end());
            if (old_order_str != new_order_str)
            {
                std::stringstream err;
                err << "Incompatible orders: old order=" << old_order.toString()
                                       << ", new order=" << new_order.toString();
                errMsg = err.str();
                return {false, 0};
            }

            return {true, 0};

        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto input = inputs[0];
            auto outputOrder = input->getOrder();
            auto old_order = input->getOrder();
            auto new_order = args.at("order").get<mv::Order>();
            auto old_order_str = old_order.toString();
            auto new_order_str = new_order.toString();
            auto cpu_in_order_str = std::string("WHCN");

            // Reverse order strings if necessary
            if (old_order_str[3] != 'N')
                old_order_str = std::string(old_order_str.rbegin(), old_order_str.rend());

            if (new_order_str[3] != 'N')
                new_order_str = std::string(new_order_str.rbegin(), new_order_str.rend());

            // Permute tensor Shape according to new Order
            // inputShape is WHCN
            // new order is permuted w.r.t. WHCN
            auto inputShape = input->getShape();
            auto ndims = inputShape.ndims();
            mv::Shape outputShape(ndims);
            for (size_t i=0; i < ndims; i++)
            {
                auto j = cpu_in_order_str.find(new_order_str[i]);
                assert(j != std::string::npos && j < ndims);
                outputShape[i] = inputShape[j];
            }

            // output tensor uses permuted shape with old order
            outputs.emplace_back(":0", outputShape, inputs[0]->getDType(), old_order);
        };

    }



    namespace op {
        // Permute:
        // Physically transpose tensor's data according to
        // the given permutation of its dimensions.
        // For example, given "NCHW" tensor of 8x3x320x200,
        // permuting last two coordinates like "NCWH" will
        // make it shaped as 8x3x200x320, but tensor order
        // would still remain "NCHW".

        // NOTE: Is this type of operation really necessary in our compiler
        // given our shape/order assumption?

        MV_REGISTER_OP(Permute)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::Order>("order")
        .setInputCheck(op_permute::inputCheckFcn)
        .setOutputDef(op_permute::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}

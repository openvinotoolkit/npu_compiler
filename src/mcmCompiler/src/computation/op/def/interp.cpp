//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_interp
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            //TODO add checks to see if outputHeight/Width both scale up or down (have the same factor?)
            auto factor = args.at("factor").get<double>();
            if (factor <= 0)
            {
                std::stringstream err;
                err << "factor must be non-zero positive: factor=" << factor;
                errMsg = err.str();
                return {false, 0};
            }
            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto factor = args.at("factor").get<double>();
            unsigned outHeight , outWidth;
            auto inputShape = inputs[0]->getShape();

            if (factor != 1)
            {
                auto inHeight = inputShape[IO_HEIGHT_DIMENSION];
                auto inWidth = inputShape[IO_WIDTH_DIMENSION];
                outHeight = inHeight * factor;
                outWidth = inWidth * factor;
            }
            else
            {
                outHeight = args.at("height").get<unsigned>();
                outWidth = args.at("width").get<unsigned>();

                if (outHeight == 0)
                {
                    auto inHeight = inputShape[IO_HEIGHT_DIMENSION];
                    auto padBeg = args.at("pad_beg").get<unsigned>();
                    auto padEnd = args.at("pad_end").get<unsigned>();
                    outHeight = inHeight + padBeg + padEnd;
                }

                if (outWidth == 0)
                {
                    auto inWidth = inputShape[IO_WIDTH_DIMENSION];
                    auto padBeg = args.at("pad_beg").get<unsigned>();
                    auto padEnd = args.at("pad_end").get<unsigned>();
                    outWidth = inWidth + padBeg + padEnd;
                }
            }

            mv::Shape outputShape({outWidth, outHeight, inputShape[IO_CHANNEL_DIMENSION], inputShape[IO_BATCH_DIMENSION]});;
            outputs.emplace_back(":0",  outputShape, inputs[0]->getDType(), inputs[0]->getOrder());
        };

    }

    namespace op {
        MV_REGISTER_OP(Interp)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<double>("factor")
        .setArg<unsigned>("pad_beg")
        .setArg<unsigned>("pad_end")
        .setOptionalArg<unsigned>("height", 0)
        .setOptionalArg<unsigned>("width", 0)
        .setOptionalArg<bool>("align_corners", true)
        .setInputCheck(op_interp::inputCheckFcn)
        .setOutputDef(op_interp::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}

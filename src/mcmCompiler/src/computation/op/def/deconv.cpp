//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/tensor/tiling.hpp"

namespace mv
{
    namespace op_deconv
    {
        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto kernel = inputs[1];
            auto kernelShape = (kernel->hasAttr("OriginalShape")) ?
                                    (kernel->get<mv::Shape>("OriginalShape")) :
                                    (kernel->getShape());

            auto data = inputs[0];
            auto dataShape = inputs[0]->getShape();

            auto group = args.at("group").get<unsigned>();
            if (group < 1)
            {
                std::stringstream err;
                err << "Group factor must be non-zero: group=" << group;
                errMsg = err.str();
                return {false, 0};
            }

            if (dataShape.ndims() != 4 )
            {
                errMsg = "Shape ndims is not equal to 4";
                return {false, 0};
            }

            if (kernelShape.ndims() != 4)
            {
                errMsg = "Shape ndims is not equal to 4";
                return {false, 1};
            }

            if (dataShape[IO_CHANNEL_DIMENSION] != kernelShape[KERNEL_INPUT_CHANNELS])
            {
                errMsg = "Number of weights input channels: " + std::to_string(kernelShape[KERNEL_INPUT_CHANNELS]) +
                " does not match input_channels: " + std::to_string(dataShape[IO_CHANNEL_DIMENSION]);
                return {false, 1};
            }

            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();

            if (dataShape[IO_WIDTH_DIMENSION] + padding[0] + padding[1] < kernelShape[KERNEL_WIDTH])
            {
                errMsg = "Width exceeds padded input width " + std::to_string(dataShape[IO_WIDTH_DIMENSION] + padding[0] + padding[1]);
                return {false, 1};
            }

            if (dataShape[IO_HEIGHT_DIMENSION] + padding[2] + padding[3] < kernelShape[KERNEL_HEIGHT])
            {
                errMsg = "Height exceeds padded input height " + std::to_string(dataShape[IO_HEIGHT_DIMENSION] + padding[2] + padding[3]);
                return {false, 1};
            }

            auto dilationFactor = args.at("dilationFactor").get<unsigned>();
            if (dilationFactor < 1) {

                errMsg = "Dilation factor must be greater than or equal to one";
                return {false, 1};

            }

            return {true, 0};
        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {

            auto kernel = inputs[1];
            auto kernelShape = (kernel->hasAttr("OriginalShape")) ?
                                (kernel->get<mv::Shape>("OriginalShape")) :
                                (kernel->getShape());

            auto data = inputs[0];
            auto dataShape = inputs[0]->getShape();

            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();
            auto stride = args.at("stride").get<std::array<unsigned short, 2>>();
            auto group = args.at("group").get<unsigned>();

            // TODO: dilation factor must be per kernel dimension
            auto dilationFactor = args.at("dilationFactor").get<unsigned>();

            auto W = stride[0] * (dataShape[IO_WIDTH_DIMENSION] - 1) + ((kernelShape[KERNEL_WIDTH] - 1) * dilationFactor + 1) - padding[0] - padding[1];
            auto H = stride[1] * (dataShape[IO_HEIGHT_DIMENSION] - 1) + ((kernelShape[KERNEL_HEIGHT] - 1) * dilationFactor + 1) - padding[2] - padding[3];
            auto C = kernelShape[KERNEL_OUTPUT_CHANNELS] * group;
            // auto C = kernelShape[KERNEL_INPUT_CHANNELS];
            auto N = dataShape[IO_BATCH_DIMENSION];

            mv::Shape outputShape({W, H, C, N});

            outputs.emplace_back(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder());
        };
    }

    namespace op {
        MV_REGISTER_OP(Deconv)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setArg<std::array<unsigned short, 2>>("stride")
        .setArg<std::array<unsigned short, 4>>("padding")
        .setOptionalArg<unsigned>("dilationFactor", 1)
        .setOptionalArg<unsigned>("group", 1)
        .setOptionalArg<bool>("is_depthwise", false)
        .setInputCheck(op_deconv::inputCheckFcn)
        .setOutputDef(op_deconv::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});
    }

}

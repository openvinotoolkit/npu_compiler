#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_slice
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            const auto& startCoord = args.at("begin").get<mv::Shape>();
            const auto& outputSize = args.at("size").get<mv::Shape>();
            const auto& inputShape = inputs[0]->getShape();

            //TODO: support variable number of dims
            if ( (startCoord.ndims() != 4) ||
                 (outputSize.ndims() != 4) ||
                 (inputShape.ndims() != 4) )
            {
                errMsg = "Invalid shape at input. InputTensor - StartingCoord - OutputSize - must have a dimensionality of 4"
                        + std::to_string(inputShape.ndims()) + " - " + std::to_string(startCoord.ndims()) + " - " + std::to_string(outputSize.ndims());
                return {false, 0};
            }

            // starting coord must be inside input tensor
            for(std::size_t dim = 0; dim < inputShape.ndims(); dim++)
            {
                auto start = startCoord[dim];
                auto size = outputSize[dim];
                auto input = inputShape[dim];

                if (start + size > input) {
                    std::ostringstream strm;
                    strm
                        << "Invalid configuration for dimension " << dim
                        << " slice [" << start << ":" << start + size << "] is ouside the input tensor " << input;

                    errMsg = strm.str();

                    return {false, 0};
                }
            }

            return {true, 0};
        };


        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            outputs.emplace_back(":0",args.at("size").get<mv::Shape>(),inputs[0]->getDType(),inputs[0]->getOrder());

            if (inputs[0]->isPopulated())
            {

                std::vector<mv::DataElement> temp(args.at("size").get<mv::Shape>().totalSize(), mv::DataElement(inputs[0]->isDoubleType()));
                outputs[0].populate(temp);

                // NOTE: Sloooooooooooooow
                auto begin = args.at("begin").get<mv::Shape>();
                auto size = args.at("size").get<mv::Shape>();
                for(unsigned oc = begin[mv::KERNEL_OUTPUT_CHANNELS]; oc < begin[mv::KERNEL_OUTPUT_CHANNELS] + size[mv::KERNEL_OUTPUT_CHANNELS]; ++oc)
                    for(unsigned ic = begin[mv::KERNEL_INPUT_CHANNELS]; ic < begin[mv::KERNEL_INPUT_CHANNELS] + size[mv::KERNEL_INPUT_CHANNELS]; ++ic)
                        for(unsigned kw = begin[mv::KERNEL_WIDTH]; kw <  begin[mv::KERNEL_WIDTH] + size[mv::KERNEL_WIDTH]; ++kw)
                            for(unsigned kh = begin[mv::KERNEL_HEIGHT]; kh < begin[mv::KERNEL_HEIGHT] + size[mv::KERNEL_HEIGHT]; ++kh)
                                outputs[0].at({kw - begin[mv::KERNEL_WIDTH],kh - begin[mv::KERNEL_HEIGHT]
                                               ,ic - begin[mv::KERNEL_INPUT_CHANNELS],oc - begin[mv::KERNEL_OUTPUT_CHANNELS]})
                                = inputs[0]->at({kw,kh,ic,oc});
            }
        };

    }

    namespace op {
        MV_REGISTER_OP(Slice)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::Shape>("begin")
        .setArg<mv::Shape>("size")
        .setInputCheck(op_slice::inputCheckFcn)
        .setOutputDef(op_slice::outputDefFcn)
        .setTypeTrait({"exposed"});
    }

}

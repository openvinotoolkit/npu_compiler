#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            auto auto_pad = args.at("auto_pad").get<std::string>();
            auto rounding_type = args.at("rounding_type").get<std::string>();

            if (auto_pad != "" && auto_pad != "same_upper" && auto_pad != "same_lower" && auto_pad != "valid")
            {
                errMsg = "Invalid argument: auto_pad=" + auto_pad;
                return {false, 0};
            }

            if (rounding_type != "floor" && rounding_type != "ceil")
            {
                errMsg = "Invalid argument: rounding_type=" + rounding_type;
                return {false, 0};
            }

            if (inputs[0]->getShape().ndims() != 4)
            {
                errMsg = "Invalid shape of the input tensor (input 0) - must have a dimensionality of 4, "
                    " has " + std::to_string(inputs[0]->getShape().ndims());
                return {false, 0};
            }

            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();
            auto kSize = args.at("kSize").get<std::array<unsigned short, 2>>();
            
            if (inputs[0]->getShape()[mv::IO_WIDTH_DIMENSION] + padding[0] + padding[1] < kSize[0])
            {
                errMsg = "Filter kernel width (" + std::to_string(kSize[0]) + ") exceeds the padded input width ("
                    + std::to_string(inputs[0]->getShape()[mv::IO_WIDTH_DIMENSION] + padding[0] + padding[1]) + ")";
                return {false, 0};
            }

            if (inputs[0]->getShape()[mv::IO_HEIGHT_DIMENSION] + padding[2] + padding[3] < kSize[1])
            {
                errMsg = "Filter kernel height (" + std::to_string(kSize[1]) + ") exceeds the padded input height ("
                    + std::to_string(inputs[0]->getShape()[mv::IO_HEIGHT_DIMENSION] + padding[2] + padding[3]) + ")";
                return {false, 0};
            }

            return {true, 0};
        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto inputShape = inputs[0]->getShape();
            auto padding = args.at("padding").get<std::array<unsigned short, 4>>();
            auto stride = args.at("stride").get<std::array<unsigned short, 2>>();
            auto kSize = args.at("kSize").get<std::array<unsigned short, 2>>();

            Shape outputShape({(inputShape[mv::IO_WIDTH_DIMENSION] + padding[0] + padding[1] - kSize[KERNEL_WIDTH]) / stride[0] + 1,
                (inputShape[mv::IO_HEIGHT_DIMENSION] + padding[2] + padding[3] - kSize[KERNEL_HEIGHT]) / stride[1] + 1, inputShape[mv::IO_CHANNEL_DIMENSION], inputShape[mv::IO_BATCH_DIMENSION]});

            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty() == true)
                outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0", outputShape, inputs[0]->getDType(), inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));

        };

        // TODO: make setOptionalArg accept "..." instead of std::string("...")
        // Default values for (some of) optional arguments
        static std::string default_auto_pad = ""; // variants: "", "same_upper", "same_lower", "valid"
        static std::string default_rounding_type = "floor"; // variants: "floor", "ceil"

        MV_REGISTER_OP(AveragePool)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<std::array<unsigned short, 2>>("kSize")
        .setArg<std::array<unsigned short, 2>>("stride")
        .setArg<std::array<unsigned short, 4>>("padding")
        .setOptionalArg<bool>("exclude_pad", true)
        .setOptionalArg<std::string>("auto_pad",      default_auto_pad)      // default: ""
        .setOptionalArg<std::string>("rounding_type", default_rounding_type) // default: "floor"
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}

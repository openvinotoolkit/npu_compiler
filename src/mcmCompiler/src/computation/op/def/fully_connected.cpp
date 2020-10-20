#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_fully_connected
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {

            if (inputs[1]->getShape().ndims() != 2)
            {
                errMsg = "Invalid shape of the weights tensor (input 1) - must have a dimensionality of 2, "
                    " has " + std::to_string(inputs[1]->getShape().ndims());
                return {false, 0};
            }

              if (inputs[0]->getShape().totalSize() != inputs[1]->getShape()[0])
            {
                errMsg = "Inconsistent total size of input tensor (input 0) " + std::to_string(inputs[0]->getShape().totalSize()) +
                    " and 1st dimension of weights tensor (input 1) " + std::to_string(inputs[1]->getShape()[0]);
                return {false, 0};
            }

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            const Shape shape = {1, 1, inputs[1]->getShape()[KERNEL_HEIGHT], 1};
            outputs.emplace_back(":0", shape, inputs[0]->getDType(), inputs[0]->getOrder());
        };

    }

    namespace op {
        MV_REGISTER_OP(FullyConnected)
        .setInputs({"data", "weights"})
        .setOutputs({"output"})
        .setInputCheck(op_fully_connected::inputCheckFcn)
        .setOutputDef(op_fully_connected::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}

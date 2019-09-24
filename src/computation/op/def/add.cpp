#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_add
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&,
            std::string& errMsg) -> std::pair<bool, std::size_t>
        {
            if (inputs[0]->getShape() != inputs[1]->getShape())
            {
                errMsg = "Does not match the data0 shape " + inputs[1]->getShape().toString();
                return {false, 1};
            }

            return {true, 0};
        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&args, std::vector<Tensor>& outputs)
        {
            if (args.at("quantParams").get<mv::QuantizationParams>().isEmpty())
                outputs.push_back(mv::Tensor(":0",  inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder()));
            else
                outputs.push_back(mv::Tensor(":0",  inputs[0]->getShape(), inputs[0]->getDType(), inputs[0]->getOrder(), args.at("quantParams").get<mv::QuantizationParams>()));
        };
    

    }

    namespace op {

        MV_REGISTER_OP(Add)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setOptionalArg<mv::QuantizationParams>("quantParams", mv::QuantizationParams({},{},{},{}))
        .setInputCheck(op_add::inputCheckFcn)
        .setOutputDef(op_add::outputDefFcn)
        .setTypeTrait({"executable", "exposed"})
        .setVariableInputNum(true);

    }

}

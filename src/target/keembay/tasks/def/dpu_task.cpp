#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/computation/op/op.hpp"

namespace mv
{

    namespace op
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args,
            std::string&) -> std::pair<bool, std::size_t>
        {
            auto opIt = args.at("taskOp").get<std::string>();
            std::string errMsg;

            //Necessary, otherwise check on attributes fails
            auto argsBackup = args;
            argsBackup.erase("taskOp");
            return mv::op::OpRegistry::checkInputs(opIt, inputs, argsBackup, errMsg);
        };
                
        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&, 
            std::vector<Tensor>&)> outputDefFcn =
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            auto opIt = args.at("taskOp").get<std::string>();

            mv::op::OpRegistry::getOutputsDef(opIt, inputs, args, outputs);
        };
    
        MV_REGISTER_OP(DPUTask)
        .setInputs({"inputs"})
        .setOutputs({"output"})
        .setArg<std::string>("taskOp")
        .setInputCheck(inputCheckFcn)
        .setInputCheck(false)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable"})
        .setVariableInputNum(true)
        .setBaseOperation({"Conv", "MaxPool"});
    }

}

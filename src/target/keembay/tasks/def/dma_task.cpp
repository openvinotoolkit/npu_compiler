#include "include/mcm/computation/op/op_registry.hpp"
#include "include/mcm/target/keembay/dma_direction.hpp"

namespace mv
{

    namespace op
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
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>&args, std::vector<Tensor>& outputs)
        {
            mv::Tensor toPush(*inputs[0]);
            outputs.push_back(std::move(toPush));
            outputs[0].setName(":0");
            outputs[0].erase("flows");

            auto direction = args.at("direction").get<mv::DmaDirection>();

            if (direction == mv::DmaDirectionEnum::DDR2NNCMX)
            {
                mv::Tensor::MemoryLocation outputLocation("NNCMX");
                outputs[0].set<mv::Tensor::MemoryLocation>("Location", outputLocation);
            }
            else if (direction == mv::DmaDirectionEnum::NNCMX2DDR ||
                     direction == mv::DmaDirectionEnum::UPACMX2DDR)
            {
                mv::Tensor::MemoryLocation outputLocation("DDR");
                outputs[0].set<mv::Tensor::MemoryLocation>("Location", outputLocation);
            }
            else if (direction == mv::DmaDirectionEnum::DDR2UPACMX ||
                     direction == mv::DmaDirectionEnum::NNCMX2UPACMX)
            {
                mv::Tensor::MemoryLocation outputLocation("UPACMX");
                outputs[0].set<mv::Tensor::MemoryLocation>("Location", outputLocation);
            }
        };

        MV_REGISTER_OP(DMATask)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::DmaDirection>("direction")
        .setInputCheck(inputCheckFcn)
        .setOutputDef(outputDefFcn)
        .setTypeTrait({"executable"});

    }

}

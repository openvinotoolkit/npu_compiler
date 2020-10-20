#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_sparsity_map
    {

        static std::function<std::pair<bool, std::size_t>(const std::vector<Data::TensorIterator>&,
            const std::map<std::string, Attribute>&, std::string&)> inputCheckFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::string&) -> std::pair<bool, std::size_t>
        {

            return {true, 0};

        };

        static std::function<void(const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>&,
            std::vector<Tensor>&)> outputIntDefFcn =
            [](const std::vector<Data::TensorIterator>&, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            outputs.emplace_back(":0", args.at("shape").get<mv::Shape>(), args.at("dType").get<mv::DType>(),
                args.at("order").get<mv::Order>(), args.at("data").get<std::vector<int64_t>>());
        };

    }

    namespace op {
        MV_REGISTER_OP(SparsityMap)
        .setOutputs({"output"})
        .setArg<std::vector<int64_t>>("data")
        .setArg<mv::Shape>("shape")
        .setArg<mv::DType>("dType")
        .setArg<mv::Order>("order")
        .setInputCheck(op_sparsity_map::inputCheckFcn)
        .setOutputDef(op_sparsity_map::outputIntDefFcn);
    }

}

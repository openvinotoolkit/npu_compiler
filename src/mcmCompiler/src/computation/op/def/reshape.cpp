#include "include/mcm/computation/op/op_registry.hpp"

namespace mv
{

    namespace op_reshape
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
            [](const std::vector<Data::TensorIterator>& inputs, const std::map<std::string, Attribute>& args, std::vector<Tensor>& outputs)
        {
            mv::Order order(inputs[0]->getOrder());

            auto new_shape = args.at("shape").get<mv::Shape>();
            if (new_shape.ndims() != 4)
            {
                new_shape = mv::Shape::augment(new_shape, 4);
            }

            outputs.emplace_back(":0",  new_shape, inputs[0]->getDType(), order);
        };

        static std::string empty;

    }

    namespace op {

        // Reshape:
        // Change tensor shape w/o physically moving data

        // Order is constant, never changes, needs an argument
        // of the outputTensorShape like tf logic

        MV_REGISTER_OP(Reshape)
        .setInputs({"data"})
        .setOutputs({"output"})
        .setArg<mv::Shape>("shape")
        .setInputCheck(op_reshape::inputCheckFcn)
        .setOutputDef(op_reshape::outputDefFcn)
        .setTypeTrait({"executable", "exposed"});

    }

}

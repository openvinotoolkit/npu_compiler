#ifndef MV_BASE_OP_MODEL_HPP_
#define MV_BASE_OP_MODEL_HPP_

#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/computation/op/op.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class BaseOpModel : public ComputationModel
    {
    	friend class CompositionalModelRecorder;
        /*bool defineDefaultControlFlow_(Data::OpListIterator op);
        bool defaultStage_(Data::OpListIterator op);*/
        
    public:

        BaseOpModel(const std::string& name);
        BaseOpModel(ComputationModel& model);
        BaseOpModel(mv::json::Value& value);
        virtual ~BaseOpModel() = 0;

        Data::OpListIterator switchContext(Control::OpListIterator other);

        Data::OpListIterator getInput();
        Data::OpListIterator getOutput();
        Data::OpListIterator opBegin() const;
        Data::OpListIterator opEnd() const;
        Data::FlowListIterator flowEnd() const;

        Data::OpListIterator getSourceOp(Data::TensorIterator tensor);
        void addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr);

        Data::TensorIterator defineOp(const std::string& opType, const std::vector<Data::TensorIterator>& inputs,
            const std::vector<std::pair<std::string, Attribute>>& args, std::string name = "", bool dpuTask = false);
        void removeOp(Data::OpListIterator op);
        Data::FlowListIterator defineFlow(Data::TensorIterator sourceTensor, Data::OpListIterator sinkOp, std::size_t inputIdx);
        Data::FlowListIterator defineFlow(Data::OpListIterator sourceOp, std::size_t outputIdx, Data::OpListIterator sinkOp, std::size_t inputIdx);
        void undefineFlow(Data::FlowListIterator flow);

        void addGroupElement(Data::OpListIterator element, GroupIterator group);
        void removeGroupElement(Data::OpListIterator element, GroupIterator group);
        using ComputationModel::addGroupElement;
        using ComputationModel::removeGroupElement;

        std::vector<Shape> getInputShapes(Data::OpListIterator op);
        std::vector<Shape> getOutputShapes(Data::OpListIterator op);

        std::size_t opsCount() const;
        std::size_t opsCount(const std::string& opType) const;

        long long unsigned parametersCount() const;

        virtual std::string getLogID() const override;

    };

    

}

#endif // MV_BASE_OP_MODEL_HPP_

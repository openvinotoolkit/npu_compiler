#include "include/mcm/computation/model/base_op_model.hpp"
#include "include/mcm/algorithms/topological_sort.hpp"
#include "include/mcm/algorithms/path_exists.hpp"
#include "include/mcm/algorithms/lexicographical_topsort.hpp"

const std::string OUT_START_TEMPL = R"cppinttempl(
// The file was generated by RecordedOpModel

#include <limits>
#include <include/mcm/op_model.hpp>
#include "include/mcm/compiler/compilation_unit.hpp"
#include "@DATA_FILE_NAME@"


template <typename T1, typename T2>
std::vector<T1>
    read(const std::string& filepath, std::size_t num = std::numeric_limits<size_t>::max())
{
    std::ifstream fileStream(filepath, std::ifstream::binary);
    if (!fileStream) 
        throw mv::RuntimeError("TemplateExample", "Weights file: \"" + filepath + "\" not found");
    std::vector<T1> data;
    T2 aux;
    while (fileStream.read(&reinterpret_cast<char&>(aux), sizeof(aux)) && num-- > 0)
        data.emplace_back(aux);
    return data;
}

void build_@MODEL_NAME@(mv::OpModel& model)
{
    using namespace mv;
    const std::string WEIGHTS_FOLDER = mv::utils::projectRootPath() + "/examples/";
    static const auto inf = std::numeric_limits<double>::infinity();

)cppinttempl";

const std::string OUT_MAIN_FUNC = R"cppinttempl(
int main()
{
    mv::CompilationUnit unit("parserModel");
    mv::OpModel& om = unit.model();
    build_@MODEL_NAME@(om);

    std::string compDescPath = mv::utils::projectRootPath() + "/config/compilation/release_kmb.json";
    unit.loadCompilationDescriptor(compDescPath);

    unit.loadTargetDescriptor(mv::Target::ma2490);
    unit.initialize();
    unit.run();

    return 0;
}
)cppinttempl";

mv::BaseOpModel::BaseOpModel(const std::string& name) :
ComputationModel(name)
{
    log(Logger::MessageType::Debug, "Initialized");
}

mv::BaseOpModel::BaseOpModel(ComputationModel& other) :
ComputationModel(other)
{
    log(Logger::MessageType::Debug, "Bound");
}

mv::BaseOpModel::~BaseOpModel()
{
    log(Logger::MessageType::Debug, "Deleted");
    // close up recorded model, if enabled
    if (codeOut_) {
        auto outMain = OUT_MAIN_FUNC;
        setTemplParam(outMain, "@MODEL_NAME@", varName(getName()));
        *codeOut_ << "}" << std::endl << outMain << std::endl;
        delete codeOut_;
        delete dataOut_;
    }
}

void mv::BaseOpModel::setTemplParam(std::string& str, const std::string& paramName, const std::string& paramValue)
{
    auto pos = std::string::npos;
    while ((pos = str.find(paramName)) != std::string::npos)
    {
        str.replace(pos, paramName.length(), paramValue);
    }
}

std::string mv::BaseOpModel::removeFileExt(const std::string& filePath)
{
    const auto pos = filePath.rfind('.');
    return pos == std::string::npos ? filePath : filePath.substr(0, pos);
}

std::string mv::BaseOpModel::varName(std::string name)
{
    std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c) && c != '_'; }, '_');
    if (!name.empty() && !std::isalpha(name[0]))
    {
        name = '_' + name;
    }
    return name;
}

void mv::BaseOpModel::initRecordingFile(const std::string& outFileName, bool recordWeightsAsText) 
{
    // log(Logger::MessageType::Debug, "Initializing RecordedModel...");
    delete codeOut_;
    delete dataOut_;

    codeOut_ = new std::ofstream();
    dataOut_ = new std::ofstream();
    this->recordModel_ = true;
    this->recordWeightsAsText_ = recordWeightsAsText;

    const auto dataFileName = removeFileExt(outFileName) + ".data.inc";
    codeOut_->open(outFileName, std::ios_base::out | std::ios_base::trunc);
    assert(codeOut_->is_open());
    dataOut_->open(dataFileName, std::ios_base::out | std::ios_base::trunc);
    assert(dataOut_->is_open());
    auto outStart = OUT_START_TEMPL;
    
    if (recordWeightsAsText) // remove the text weights file include if not required
        setTemplParam(outStart, "@DATA_FILE_NAME@", dataFileName);
    else
        setTemplParam(outStart, "#include \"@DATA_FILE_NAME@\"", "");
    setTemplParam(outStart, "@MODEL_NAME@", varName(getName()));
    *codeOut_ << outStart;
}

mv::Data::OpListIterator mv::BaseOpModel::switchContext(Control::OpListIterator other)
{
    return opsGraph_->get_first_iterator(other);
}

mv::Data::OpListIterator mv::BaseOpModel::getSourceOp(Data::TensorIterator tensor)
{

    if (!tensor->hasAttr("sourceOp"))
        return opEnd();

    auto it = ops_->find(tensor->get<std::string>("sourceOp"));
    if (it == ops_->end())
        throw RuntimeError(*this, "Source op " + tensor->get<std::string>("sourceOp") + " of tensor " +
            tensor->getName() + " does not belong to the model");

    return it->second;

}

mv::Data::TensorIterator mv::BaseOpModel::defineOp(const std::string& opType, const std::vector<Data::TensorIterator>& inputs,
            const std::vector<std::pair<std::string, Attribute>> & args, std::string name, bool checkInputSize, bool checkArgs)
{

    if (name.empty())
    {
        if (opsIndexCounter_->find(opType) != opsIndexCounter_->end())
            name = opType + "_" + std::to_string(opsIndexCounter_->at(opType));
        else
            name = opType + "_0";
    }

    if (ops_->find(name) != ops_->end())
        throw ArgumentError(*this, "op:name", name, "Duplicated op name");

    auto opNode = dataGraph_.node_insert(Op(*this, opType, name, inputs, args, checkInputSize, checkArgs));

    incrementOpsInstanceCounter_(opType);
    incrementOpsIndexCounter_(opType);

    ops_->emplace(name, opNode);

    for (std::size_t i = 0; i < (*opNode).inputSlots(); ++i)
        defineFlow(inputs[i], opNode, i);

    log(Logger::MessageType::Debug, "Defined " + (*opNode).toString());

    // Assumes single input/output
    if (opType == "Input")
    {
        bool networkInput = false;
        for (auto arg: args)
        {
            if (arg.first == "networkInput")
            {
                networkInput = arg.second;
            }
        }

        if (getNumNetworkInputs() && !networkInput)
        {
            *input_ = opNode;
        }
        else
        {
            networkInputs_->push_back(opNode);
        }

    }
    else if (opType == "Output")
    {
        bool networkOutput = false;
        for (auto arg: args)
        {
            if (arg.first == "networkOutput")
            {
                networkOutput = arg.second;
            }
        }

        if (getNumNetworkOutputs() && !networkOutput)
        {
            *output_ = opNode;
        }
        else
        {
            networkOutputs_->push_back(opNode);
        }
    }

    if ((*opNode).outputSlots() > 0)
        return (*opNode).getOutputTensor(0);

    return tensorEnd();

}

void mv::BaseOpModel::removeOp(Data::OpListIterator op)
{

    if (op == opEnd())
        throw ArgumentError(*this, "op:iterator", "end", "Invalid iterator passed for op removal");

    //Removing input/output data flows from the model
    //There is no actual need to call undefineFlow, as the graph structure will be handled by dataGraph_.node_erase(op)
    //But undefineFlow also removes the flow information from the tensor, so it's better to use it

    // Apparentely this iterator is missing an input flow sometimes
    auto sourceFlow = op.leftmostInput();
    while (sourceFlow != flowEnd())
    {
        auto backup = sourceFlow;
        ++sourceFlow;
        undefineFlow(backup);
        //dataFlows_->erase(sourceFlow->getName());
    }

    auto sinkFlow = op.leftmostOutput();
    while (sinkFlow != flowEnd())
    {
        auto backup = sinkFlow;
        ++sinkFlow;
        undefineFlow(backup);
    }

    //Removing output tensors from the model
    for (std::size_t j = 0; j < op->outputSlots(); ++j)
        tensors_->erase(op->getOutputTensor(j)->getName());

    decrementOpsInstanceCounter_(op->getOpType());
    ops_->erase(op->getName());

    log(Logger::MessageType::Debug, "Removed " + op->toString());
    dataGraph_.node_erase(op);

}

mv::Data::FlowListIterator mv::BaseOpModel::defineFlow(Data::TensorIterator sourceTensor, Data::OpListIterator sinkOp, std::size_t inputIdx)
{

    if (!isValid(sourceTensor))
        throw ArgumentError(*this, "sourceTensor", "invalid", "Invalid tensor passed for the data flow definition");

    if (!isValid(sinkOp))
        throw ArgumentError(*this, "sinkOp", "invalid", "Invalid sink op passed for the data flow definition");

    auto sourceOp = getSourceOp(sourceTensor);
    if (sourceOp == opEnd())
        throw ArgumentError(*this, "sourceTensor", "sourceless", "Defining flow using a tensor that does not have a source op is illegal");

    Data::FlowListIterator inputFlow = dataGraph_.edge_insert(sourceOp, sinkOp, DataFlow(*this, sourceOp, 0, sinkOp, inputIdx, sourceTensor));

    if(!sourceTensor->hasAttr("flows"))
    {
        std::set<std::string> toSet;
        sourceTensor->set<std::set<std::string>>("flows", toSet);
    }

    sourceTensor->get<std::set<std::string>>("flows").insert(inputFlow->getName());
    dataFlows_->emplace(inputFlow->getName(), inputFlow);
    log(Logger::MessageType::Debug, "Defined " + inputFlow->toString());
    return inputFlow;

}

bool mv::BaseOpModel::pathExists(Data::OpListIterator source, Data::OpListIterator target)
{
    return mv::pathExists(dataGraph_, source, target);
}


mv::Data::FlowListIterator mv::BaseOpModel::defineFlow(Data::OpListIterator sourceOp, std::size_t outputIdx, Data::OpListIterator sinkOp, std::size_t inputIdx)
{

    auto sourceTensor = sourceOp->getOutputTensor(outputIdx);
    return defineFlow(sourceTensor, sinkOp, inputIdx);

}

std::vector<mv::Data::OpListIterator> mv::BaseOpModel::topologicalSort()
{
    // Necessary for correct iterator casting
    auto topologicalSortResult = mv::topologicalSort(dataGraph_);
    std::vector<mv::Data::OpListIterator> toReturn(topologicalSortResult.begin(), topologicalSortResult.end());
    return toReturn;
}

struct OpItComparator
{
    bool operator()(mv::Data::OpListIterator lhs, mv::Data::OpListIterator rhs) const
    {
        return (lhs->getName() < rhs->getName());
    }
};

struct OpLexComparator
{
    bool operator()(mv::Data::OpListIterator lhs, mv::Data::OpListIterator rhs) const
    {
        return !(lhs->getName() < rhs->getName());
    }
};

std::vector<mv::Data::OpListIterator> mv::BaseOpModel::lexTopologicalSort()
{
    auto lexTopSortResult = mv::lexTopologicalSort<Op, DataFlow, OpItComparator, OpLexComparator>(dataGraph_);
    std::vector<mv::Data::OpListIterator> toReturn(lexTopSortResult.begin(), lexTopSortResult.end());

    log(Logger::MessageType::Debug, "LexTopological Sorted Operations: ");
    for (auto s: toReturn)
    {
        log(Logger::MessageType::Debug, s->getName());
    }
    return toReturn;
}

void mv::BaseOpModel::undefineFlow(Data::FlowListIterator flow)
{

    if (!ComputationModel::isValid(flow))
        throw ArgumentError(*this, "flow:iterator", "invalid", "Invalid flow passed for deletion");

    log(Logger::MessageType::Debug, "Removed " + flow->toString());

    if(!flow->getTensor()->hasAttr("flows"))
        log(Logger::MessageType::Error, flow->getTensor()->getName() + " is in a flow but has no attribute flows");

    flow->getTensor()->get<std::set<std::string>>("flows").erase(flow->getName());
    dataFlows_->erase(flow->getName());
    dataGraph_.edge_erase(flow);

}

mv::Data::OpListIterator mv::BaseOpModel::getInput()
{
    return *input_;
}

mv::Data::OpListIterator mv::BaseOpModel::getOutput()
{
    return *output_;
}

std::vector<mv::Data::OpListIterator> mv::BaseOpModel::getNetworkInputs()
{
    return *networkInputs_;
}

mv::Data::OpListIterator mv::BaseOpModel::getNetworkInput(std::size_t idx)
{
    if (idx >= networkInputs_->size())
        throw ArgumentError(*this, "baseOpModel", "invalid", "Network input index out of range");

    return (*networkInputs_)[idx];
}

size_t mv::BaseOpModel::getNumNetworkInputs()
{
    return networkInputs_->size();
}

void mv::BaseOpModel::setInputNode(Data::OpListIterator input)
{
    if (!input)
        throw ArgumentError(*this, "baseOpModel", "invalid", "input argument is null");

    *input_ = input;
}

void mv::BaseOpModel::replaceNetworkInputAtIdx(std::size_t idx, mv::Data::OpListIterator replacementOp)
{
    if (!replacementOp)
        throw ArgumentError(*this, "baseOpModel", "invalid", "Input argument is null");

    if (idx >= networkInputs_->size())
        throw ArgumentError(*this, "baseOpModel", "invalid", "Invalid network input index");

    (*networkInputs_)[idx] = replacementOp;
}

std::vector<mv::Data::OpListIterator> mv::BaseOpModel::getNetworkOutputs()
{
    return *networkOutputs_;
}

mv::Data::OpListIterator mv::BaseOpModel::getNetworkOutput(std::size_t idx)
{
    if (idx >= networkOutputs_->size())
        throw ArgumentError(*this, "baseOpModel", "invalid", "Network output index out of range");

    return (*networkOutputs_)[idx];
}

size_t mv::BaseOpModel::getNumNetworkOutputs()
{
    return networkOutputs_->size();
}

void mv::BaseOpModel::setOutputNode(Data::OpListIterator output)
{
    if (!output)
        throw ArgumentError(*this, "baseOpModel", "invalid", "input argument is null");

    *output_ = output;
}

void mv::BaseOpModel::replaceNetworkOutputAtIdx(std::size_t idx, mv::Data::OpListIterator replacementOp)
{
    if (!replacementOp)
        throw ArgumentError(*this, "baseOpModel", "invalid", "Input argument is null");

    if (idx >= networkOutputs_->size())
        throw ArgumentError(*this, "baseOpModel", "invalid", "Invalid network output index");

    (*networkOutputs_)[idx] = replacementOp;
}

mv::Data::OpListIterator mv::BaseOpModel::opBegin() const
{
    return dataGraph_.node_begin();
}

mv::Data::OpListIterator mv::BaseOpModel::opEnd() const
{
    return *dataOpEnd_;
}

mv::Data::FlowListIterator mv::BaseOpModel::flowEnd() const
{
    return *dataFlowEnd_;
}

void mv::BaseOpModel::addGroupElement(Data::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while including op to a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while including op to a group");

    group->include(element);
}

void mv::BaseOpModel::removeGroupElement(Data::OpListIterator element, GroupIterator group)
{
    if (!isValid(element))
        throw ArgumentError(*this, "newElement:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    if (!isValid(group))
        throw ArgumentError(*this, "group:iterator", "invalid", "Invalid iterator passed while excluding op from a group");
    group->exclude(element);
}

std::vector<mv::Shape> mv::BaseOpModel::getInputShapes(Data::OpListIterator op)
{

    if (!isValid(op))
        throw ArgumentError(*this, "op", "invalid", "Invalid op iterator passed getting inputs shapes");

    std::vector<Shape> shapes;
    for (auto it = op.leftmostInput(); it != *dataFlowEnd_; ++it)
        shapes.push_back(it->getTensor()->getShape());
    return shapes;

}

std::vector<mv::Shape> mv::BaseOpModel::getOutputShapes(Data::OpListIterator op)
{

    if (!isValid(op))
        throw ArgumentError(*this, "op", "invalid", "Invalid op iterator passed getting outputs shap");

    std::vector<Shape> shapes;
    for (auto it = op.leftmostOutput(); it != *dataFlowEnd_; ++it)
        shapes.push_back(it->getTensor()->getShape());

    return shapes;

}

std::size_t mv::BaseOpModel::opsCount() const
{
    return dataGraph_.node_size();
}

std::size_t mv::BaseOpModel::opsCount(const std::string& opType) const
{
    if (opsInstanceCounter_->find(opType) != opsInstanceCounter_->end())
        return opsInstanceCounter_->at(opType);
    return 0;
}

std::size_t mv::BaseOpModel::dataFlowsCount() const
{
    return dataGraph_.edge_size();
}

long long unsigned mv::BaseOpModel::parametersCount() const
{

    unsigned result = 0;

    for (auto it = *input_; it != opEnd(); ++it)
    {
        if (it->getOpType() == "Constant")
        {
            result += it->getOutputTensor(0)->getShape().totalSize();
        }
    }

    return result;

}

void mv::BaseOpModel::addAttr(Data::OpListIterator op, const std::string& name, const Attribute& attr)
{
    op->set(name, attr);
}

void mv::BaseOpModel::eraseAttr(Data::OpListIterator op, const std::string& name)
{
    op->erase(name);
}

std::string mv::BaseOpModel::getLogID() const
{
    return "OpModel:" + name_;
}

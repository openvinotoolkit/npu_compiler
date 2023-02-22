//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/dot_graph_writer.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4267)  // size_t to integer conversion
#endif

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/GraphWriter.h>
#include <llvm/Support/raw_ostream.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace vpux;

namespace {

constexpr size_t MAX_EDGE_NUM = 64;
constexpr size_t MAX_ATTR_STR_SIZE = 80;

enum class EdgeDir { EDGE_SKIP, EDGE_NORMAL, EDGE_REVERSE };

// This is a optimized for VPUX copy of the LLVM GraphWriter (llvm/Support/GraphWriter.h).
class GraphWriter final {
public:
    GraphWriter(mlir::Block& block, llvm::raw_ostream& os, const GraphWriterParams& params)
            : _block(block), _os(os), _params(params) {
    }

public:
    void writeGraph() {
        writeHeader();
        writeNodes();
        writeFooter();
    }

private:
    void writeHeader(StringRef title = {});
    void writeNodes();

    void writeFooter() {
        _os << "}\n";
    }

private:
    static EdgeDir getEdgeDirection(mlir::Operation* source, mlir::Operation* target);
    bool isNodeHidden(mlir::Operation* op) const;
    static bool isTaskNode(mlir::Operation* op);
    static bool printNodeAttributes(mlir::Operation* op, llvm::raw_ostream& os);
    std::string getNodeLabel(mlir::Operation* op);
    void writeNode(mlir::Operation* op);
    void writeEdges(mlir::Operation* op);
    void writeEdge(mlir::Operation* source, mlir::Operation* target);
    void emitEdge(const void* sourceID, const void* targetID);

private:
    mlir::Block& _block;
    llvm::raw_ostream& _os;
    GraphWriterParams _params;

    void appendShapeLabel(llvm::raw_string_ostream& os, ArrayRef<int64_t> shape, const mlir::Type& type,
                          const DimsOrder& dims) const;
};

void GraphWriter::writeHeader(StringRef title) {
    if (title.empty()) {
        _os << "digraph unnamed {\n";
    } else {
        const auto titleStr = title.str();
        _os << "digraph \"" << llvm::DOT::EscapeString(titleStr) << "\" {\n";
        _os << "\tlabel=\"" << llvm::DOT::EscapeString(titleStr) << "\";\n";
    }

    _os << "\n";
}

void GraphWriter::writeNodes() {
    llvm::DenseSet<mlir::Operation*> processedNodes;

    bool generating = _params.startAfter.empty();
    size_t subGraphId = 0;

    const auto writeNodesAndEdges = [&](mlir::Operation* op) {
        if (!isNodeHidden(op) && !processedNodes.contains(op)) {
            writeNode(op);
            writeEdges(op);
            processedNodes.insert(op);
        }
    };

    for (auto& op : _block) {
        const auto opName = vpux::stringifyLocation(op.getLoc());

        if (generating) {
            if (op.getNumRegions() == 0) {
                writeNodesAndEdges(&op);
            } else if (!processedNodes.contains(&op)) {
                _os << "subgraph cluster_" << subGraphId++ << " {\n\tstyle=filled;color=beige;\n";
                writeNode(&op);

                for (auto& region : op.getRegions()) {
                    for (auto& block : region.getBlocks()) {
                        for (auto& innerOp : block) {
                            writeNodesAndEdges(&innerOp);
                        }
                    }
                }

                _os << "}\n";

                writeEdges(&op);
                processedNodes.insert(&op);
            }

            for (auto operand : op.getOperands()) {
                if (auto* producer = operand.getDefiningOp()) {
                    writeNodesAndEdges(producer);
                }
            }
        }

        if (isTaskNode(&op)) {
            if (!_params.startAfter.empty() && (opName == _params.startAfter)) {
                generating = true;
            }
            if (!_params.stopBefore.empty() && (opName == _params.stopBefore)) {
                break;
            }
        }
    }
}  // namespace

EdgeDir GraphWriter::getEdgeDirection(mlir::Operation* source, mlir::Operation* target) {
    auto sideEffectsOp = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(target);
    if (sideEffectsOp == nullptr) {
        return EdgeDir::EDGE_NORMAL;
    }

    const auto targetOperands = target->getOperands();
    const auto it = llvm::find_if(targetOperands, [&](mlir::Value arg) {
        return arg.getDefiningOp() == source;
    });
    VPUX_THROW_UNLESS(it != targetOperands.end(), "Wrong edge");

    const auto val = *it;
    if (sideEffectsOp.getEffectOnValue<mlir::MemoryEffects::Write>(val).hasValue()) {
        return EdgeDir::EDGE_REVERSE;
    }

    return EdgeDir::EDGE_NORMAL;
}

bool GraphWriter::isNodeHidden(mlir::Operation* op) const {
    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return true;
    }

    if (op->hasTrait<mlir::OpTrait::ConstantLike>() && !_params.printConst) {
        return true;
    }

    const auto isDeclarationOp = [](mlir::Operation* op) {
        if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
            return false;
        }

        return op->hasTrait<DeclarationOp>() || mlir::isa<mlir::memref::AllocOp>(op);
    };

    if (isDeclarationOp(op) && !_params.printDeclarations) {
        return true;
    }

    return false;
}

bool GraphWriter::isTaskNode(mlir::Operation* op) {
    if (mlir::isa<mlir::memref::AllocOp>(op) || mlir::isa<mlir::memref::DeallocOp>(op)) {
        return false;
    }

    if (op->hasTrait<DeclarationOp>() || op->hasTrait<mlir::OpTrait::ConstantLike>() ||
        op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return false;
    }

    return true;
}

bool GraphWriter::printNodeAttributes(mlir::Operation* op, llvm::raw_ostream& os) {
    if (auto dotInterface = mlir::dyn_cast<DotInterface>(op)) {
        switch (dotInterface.getNodeColor()) {
        case DotNodeColor::RED:
            os << " style=filled, fillcolor=red";
            return true;
        case DotNodeColor::GREEN:
            os << " style=filled, fillcolor=green";
            return true;
        case DotNodeColor::AQUA:
            os << " style=filled, fillcolor=cyan";
            return true;
        case DotNodeColor::BLUE:
            os << " style=filled, fillcolor=blue";
            return true;
        case DotNodeColor::ORANGE:
            os << " style=filled, fillcolor=orange";
            return true;
        case DotNodeColor::AQUAMARINE:
            os << " style=filled, fillcolor=aquamarine";
            return true;
        default:
            return false;
        }
    }

    return false;
}

const std::string htmlEncode(StringRef data) {
    std::string buffer;
    buffer.reserve(data.size());
    for (size_t pos = 0; pos != data.size(); ++pos) {
        switch (data[pos]) {
        case '&':
            buffer.append("&amp;");
            break;
        case '\"':
            buffer.append("&quot;");
            break;
        case '\'':
            buffer.append("&apos;");
            break;
        case '<':
            buffer.append("&lt;");
            break;
        case '>':
            buffer.append("&gt;");
            break;
        default:
            buffer.push_back(data[pos]);
            break;
        }
    }
    return buffer;
}

std::string GraphWriter::getNodeLabel(mlir::Operation* op) {
    std::string ostr;
    llvm::raw_string_ostream os(ostr);

    auto htmlBegin = "<tr><td align='left'><font point-size='11.0'>";
    auto htmlMiddle = " </font></td>\n<td align='right'><font point-size='11.0'>";
    auto htmlEnd = " </font></td></tr>\n";
    if (_params.htmlLike) {
        os << "<tr><td align='center' colspan='2'><font point-size='14.0'><b>" << op->getName() << "</b>" << htmlEnd;
    } else {
        os << op->getName() << "\n";
    }

    if (auto dotInterface = mlir::dyn_cast<DotInterface>(op)) {
        std::string temp_str;
        llvm::raw_string_ostream temp_os(temp_str);
        // In case Operation implements custom attribute printer skip default attributes printing
        if (dotInterface.printAttributes(temp_os)) {
            if (_params.htmlLike) {
                os << htmlBegin << temp_str << htmlEnd;
            } else {
                os << temp_str << "\n";
            }
            return os.str();
        }
    }

    if (!mlir::isa<mlir::async::ExecuteOp>(op)) {
        if (_params.htmlLike) {
            os << htmlBegin << "Name:" << htmlMiddle << htmlEncode(stringifyLocation(op->getLoc())) << htmlEnd;
            os << htmlBegin << "Type:" << htmlMiddle;
        } else {
            os << stringifyLocation(op->getLoc()) << "\n";
        }

        // Print resultant types
        for (const auto type : op->getResultTypes()) {
            if (const auto ndType = type.dyn_cast<vpux::NDTypeInterface>()) {
                appendShapeLabel(os, ndType.getShape().raw(), ndType.getElementType(), ndType.getDimsOrder());
                if (const auto memref = type.dyn_cast<mlir::MemRefType>()) {
                    if (memref.getMemorySpace() != nullptr) {
                        os << " at " << memref.getMemorySpace();
                    }
                }
            } else {
                std::string temp_str;
                llvm::raw_string_ostream temp_os(temp_str);
                temp_os << type;
                if (_params.htmlLike) {
                    temp_str = htmlEncode(temp_str);
                }
                os << temp_str;
            }

            os << ", ";
        }

        if (_params.htmlLike) {
            os << htmlEnd;
        } else {
            os << "\n";
        }
    }

    for (auto& attr : op->getAttrs()) {
        if (_params.htmlLike) {
            os << htmlBegin << attr.getName() << ": ";
            os << htmlMiddle;
        } else {
            os << '\n' << attr.getName() << ": ";
        }

        if (const auto map = attr.getValue().dyn_cast<mlir::AffineMapAttr>()) {
            DimsOrder::fromAffineMap(map.getValue()).printFormat(os);
        } else {
            std::string temp_str;
            llvm::raw_string_ostream temp_os(temp_str);
            attr.getValue().print(temp_os);
            if (_params.htmlLike) {
                temp_str = htmlEncode(temp_str);
            }

            if (temp_str.size() < MAX_ATTR_STR_SIZE) {
                os << temp_str;
            } else {
                os << "[...]";
            }
        }

        if (_params.htmlLike) {
            os << htmlEnd;
        }
    }

    return os.str();
}

void GraphWriter::appendShapeLabel(llvm::raw_string_ostream& os, ArrayRef<int64_t> shape, const mlir::Type& type,
                                   const DimsOrder& dims) const {
    for (auto dim : shape) {
        if (mlir::ShapedType::isDynamic(dim)) {
            os << '?';
        } else {
            os << dim;
        }
        os << 'x';
    }
    std::string temp_str;
    llvm::raw_string_ostream temp_os(temp_str);
    type.print(temp_os);
    if (_params.htmlLike) {
        temp_str = htmlEncode(temp_str);
    }
    if (temp_str.size() < MAX_ATTR_STR_SIZE) {
        os << temp_str;
    } else {
        os << temp_str.substr(0, MAX_ATTR_STR_SIZE) << "[...]";
    }
    os << "#";
    dims.printFormat(os);
}

void GraphWriter::writeNode(mlir::Operation* op) {
    _os << "\tNode" << static_cast<const void*>(op) << " [shape=box,";

    if (printNodeAttributes(op, _os)) {
        _os << ",";
    }

    if (_params.htmlLike) {
        _os << " label=<<TABLE BORDER=\"0\" CELLPADDING=\"0\" CELLSPACING=\"0\">";
        _os << getNodeLabel(op);
        _os << "</TABLE>>];\n";
    } else {
        _os << "label=\"{";
        _os << llvm::DOT::EscapeString(getNodeLabel(op));
        _os << "}\"];\n";
    }
}

void GraphWriter::writeEdges(mlir::Operation* op) {
    for (auto* target : op->getUsers()) {
        if (!isNodeHidden(target)) {
            writeEdge(op, target);
        }
    }
}

void GraphWriter::writeEdge(mlir::Operation* source, mlir::Operation* target) {
    switch (getEdgeDirection(source, target)) {
    case EdgeDir::EDGE_NORMAL:
        emitEdge(static_cast<const void*>(source), static_cast<const void*>(target));
        break;
    case EdgeDir::EDGE_REVERSE:
        emitEdge(static_cast<const void*>(target), static_cast<const void*>(source));
        break;
    default:
        break;
    }
}

void GraphWriter::emitEdge(const void* sourceID, const void* targetID) {
    _os << "\tNode" << sourceID;
    _os << " -> Node" << targetID;
    _os << ";\n";
}

}  // namespace

mlir::LogicalResult vpux::writeDotGraph(mlir::Block& block, StringRef fileName, const GraphWriterParams& params) {
    int FD = -1;
    const auto EC =
            llvm::sys::fs::openFileForWrite(fileName, FD, llvm::sys::fs::CD_CreateAlways, llvm::sys::fs::OF_Text);
    if ((FD == -1) || (EC && EC != std::errc::file_exists)) {
        llvm::errs() << "Error writing into file " << fileName << "\n";
        return mlir::failure();
    }

    llvm::raw_fd_ostream os(FD, /*shouldClose=*/true);

    GraphWriter writer(block, os, params);
    writer.writeGraph();

    return mlir::success();
}

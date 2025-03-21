//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/json_utils.hpp"
#include "vpux/compiler/utils/strings.hpp"

#include <fstream>
#include <iomanip>

namespace vpux {
namespace VPU {

llvm::Expected<llvm::json::Value> readManualStrategyJSON(StringRef fileName) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for input strategy json was not provided");

    std::ifstream i(fileName.str());
    VPUX_THROW_UNLESS(i.good(), "File with manual strategy not opened correctly");
    std::stringstream input{};
    input << i.rdbuf();
    return llvm::json::parse(input.str());
}

void writeManualStrategyJSON(StringRef fileName, const llvm::json::Value& json) {
    VPUX_THROW_WHEN(fileName.empty(), "Output file name for output strategy json was not provided");

    std::ofstream os(fileName.str());
    VPUX_THROW_UNLESS(os.good(), "File with manual strategy not created correctly");
    os << llvm::formatv("{0:2}", json).str() << std::endl;

    return;
}

llvm::json::Value convertAttrToJSON(mlir::Attribute attr) {
    if (attr.isa<VPU::MultiClusterStrategyAttr>()) {
        return stringifyMultiClusterStrategy(attr.cast<VPU::MultiClusterStrategyAttr>().getValue());
    } else if (attr.isa<mlir::ArrayAttr>()) {
        auto values = Shape(parseIntArrayAttr<int64_t>(attr.cast<mlir::ArrayAttr>()));
        VPUX_THROW_UNLESS(values.size() == 4, "Shape has fewer dimensions than expected (4), got '{0}'", values.size());
        llvm::json::Object tilingStrategy{};
        tilingStrategy["N"] = values[Dims4D::Act::N];
        tilingStrategy["C"] = values[Dims4D::Act::C];
        tilingStrategy["H"] = values[Dims4D::Act::H];
        tilingStrategy["W"] = values[Dims4D::Act::W];

        return tilingStrategy;
    }
    VPUX_THROW("Conversion from this attribute '{0}' to string not implemented", attr);
}

mlir::Attribute convertJSONToAttr(mlir::Attribute oldAttr, const llvm::json::Value& newAttrVal) {
    if (oldAttr.isa<VPU::MultiClusterStrategyAttr>()) {
        return VPU::MultiClusterStrategyAttr::get(
                oldAttr.getContext(), symbolizeMultiClusterStrategy(newAttrVal.getAsString().value()).value());
    } else if (oldAttr.isa<mlir::ArrayAttr>()) {
        Shape newShape(4, 1);
        VPUX_THROW_WHEN(newAttrVal.getAsObject() == nullptr, "Invalid JSON representation of array attribute");
        llvm::json::Object dimenstions = *newAttrVal.getAsObject();
        newShape[Dims4D::Act::N] = static_cast<int64_t>(dimenstions["N"].getAsUINT64().value());
        newShape[Dims4D::Act::C] = static_cast<int64_t>(dimenstions["C"].getAsUINT64().value());
        newShape[Dims4D::Act::H] = static_cast<int64_t>(dimenstions["H"].getAsUINT64().value());
        newShape[Dims4D::Act::W] = static_cast<int64_t>(dimenstions["W"].getAsUINT64().value());
        return getIntArrayAttr(oldAttr.getContext(), newShape);
    }
    VPUX_THROW("Conversion from this attribute '{0}' to string not implemented", oldAttr);
}

std::optional<llvm::json::Value> getPreviousAttributeValue(const llvm::json::Value& json, const std::string& opName,
                                                           StringRef attribute) {
    auto jsonAsObject = json.getAsObject();
    if (jsonAsObject == nullptr) {
        return std::nullopt;
    }

    if (jsonAsObject->find(opName) == jsonAsObject->end()) {
        return std::nullopt;
    }

    auto jsonOpsToAttributes = *jsonAsObject;
    if (jsonOpsToAttributes[opName].getAsObject() == nullptr) {
        return std::nullopt;
    }

    auto jsonAttrsToLayerAttribute = *jsonOpsToAttributes[opName].getAsObject();
    if (jsonAttrsToLayerAttribute.find(attribute.str()) != jsonAttrsToLayerAttribute.end()) {
        return jsonAttrsToLayerAttribute[attribute.str()];
    }

    return std::nullopt;
}

std::string getOpHash(mlir::Operation* op) {
    if (op == nullptr) {
        return "Null";
    }
    std::string opLocation;
    std::hash<std::string> hasher;
    llvm::raw_string_ostream oLocation(opLocation);

    op->getLoc().print(oLocation);

    std::stringstream hexHash;
    hexHash << "0x" << std::setw(4) << std::setfill('0') << std::hex << hasher(opLocation);
    return hexHash.str();
}

void createStrategyJSONFromOperations(llvm::json::Value& json,
                                      llvm::MapVector<mlir::Location, mlir::Operation*>& operations,
                                      DenseMap<StringRef, StringRef>& strategyAttributes) {
    llvm::json::Object opsToStrategies{};
    for (auto& op : operations) {
        auto opName = vpux::stringifyPrimaryLocation(op.first);
        auto parentClusterOp = op.second->getParentOfType<VPU::NCEClusterTilingOp>();
        auto parentVFOp = op.second->getParentOfType<VPU::VerticalFusionOp>();

        // retrieve related attributes and save in JSON
        llvm::json::Object layerAttributes{};
        bool updatedVFTiling = false;
        for (const auto& attribute : strategyAttributes) {
            llvm::json::Value attributeValue(attribute.second);
            if (op.second->hasAttr(attribute.first)) {
                // Get value present in IR
                attributeValue = convertAttrToJSON(op.second->getAttr(attribute.first));
            } else if (parentClusterOp != nullptr && parentClusterOp->hasAttr(attribute.first)) {
                // Get value from parentClusterOp
                attributeValue = convertAttrToJSON(parentClusterOp->getAttr(attribute.first));
            } else {
                // If opName is found, assign the value read from previous runs
                attributeValue = getPreviousAttributeValue(json, opName, attribute.first).value_or(attributeValue);
            }

            if (attribute.first == vpux::layerTypeName) {
                std::string layerTypeName;
                llvm::raw_string_ostream oLayerTypeName(layerTypeName);
                op.second->getName().print(oLayerTypeName);
                attributeValue = std::move(layerTypeName);
            }
            if (parentVFOp != nullptr && attribute.first == vpux::tilingStrategy) {
                // If such layer is encountered, we can no longer find the tilingStrategy in NCEOp
                auto prevAttrValue = getPreviousAttributeValue(json, opName, attribute.first).value_or("");
                auto currentAttrValue = convertAttrToJSON(parentVFOp->getAttr(attribute.first));
                attributeValue = currentAttrValue;
                updatedVFTiling = prevAttrValue != currentAttrValue;
            }

            layerAttributes[attribute.first.str()] = std::move(attributeValue);
        }
        layerAttributes[vpux::verticalFusion] = parentVFOp != nullptr ? "True" : "False";
        layerAttributes[vpux::verticalFusionHash] = getOpHash(parentVFOp);
        layerAttributes[vpux::updatedVFTiling] = updatedVFTiling ? "True" : "False";
        opsToStrategies[std::move(opName)] = llvm::json::Value(std::move(layerAttributes));
    }
    json = llvm::json::Value(std::move(opsToStrategies));
}

void overwriteManualStrategy(llvm::json::Value& manualStrategyValue,
                             llvm::MapVector<mlir::Location, mlir::Operation*>& operations) {
    DenseMap<mlir::Operation*, std::pair<llvm::json::Value, bool>> vfOpVisited;
    SmallVector<StringRef> allowedValues = {layerTypeName, verticalFusionHash, verticalFusion, updatedVFTiling};
    auto isAllowedAttr = [&allowedValues](std::string& currentType) {
        return std::find(allowedValues.begin(), allowedValues.end(), currentType) != allowedValues.end();
    };

    for (auto& op : operations) {
        const auto opName = vpux::stringifyPrimaryLocation(op.first);

        VPUX_THROW_WHEN(manualStrategyValue.getAsObject() == nullptr,
                        "Manual strategy JSON should represent JSON object");
        llvm::json::Object manualStrategyObject = *manualStrategyValue.getAsObject();
        // check if manual strategy for layer exists
        if (manualStrategyObject.find(opName) == manualStrategyObject.end()) {
            continue;
        }

        auto parentVerticalFusionOp = op.second->getParentOfType<VPU::VerticalFusionOp>();
        VPUX_THROW_WHEN(manualStrategyObject[opName].getAsObject() == nullptr,
                        "JSON value for operation should represent JSON object");
        auto currOpStrategyObject = *manualStrategyObject[opName].getAsObject();
        for (auto it = currOpStrategyObject.begin(); it != currOpStrategyObject.end(); ++it) {
            // replace attributes of the operation (skip NONE) using it->second
            if (!(it->second.kind() == llvm::json::Value::Kind::String) ||
                (it->second.kind() == llvm::json::Value::Kind::String &&
                 it->second.getAsString().value() != defaultNoValue.str())) {
                if (it->first.str() == multiClusterStrategy) {
                    // Clustering is set as placeholder to be replaced with provided strategy
                    auto dummyAttr = VPU::MultiClusterStrategyAttr::get(op.second->getContext(),
                                                                        VPU::MultiClusterStrategy::Clustering);
                    auto manualAttribute = convertJSONToAttr(dummyAttr, it->second);

                    if (auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(op.second)) {
                        auto manualStratAttr = manualAttribute.cast<VPU::MultiClusterStrategyAttr>();
                        clusteredOp.setMultiClusterStrategy(manualStratAttr.getValue());
                    }

                } else if (it->first.str() == tilingStrategy) {
                    // tiling case, where strategy selection and IR modification occurs in a single pass
                    // TODO: remove "else" when tiling strategy will be abstracted into strategy pass
                    auto dummyAttr = getIntArrayAttr(op.second->getContext(), Shape(4));
                    auto manualAttribute = convertJSONToAttr(dummyAttr, it->second);
                    if (parentVerticalFusionOp != nullptr) {
                        auto [itr, inserted] = vfOpVisited.try_emplace(parentVerticalFusionOp, it->second, true);
                        if (!inserted) {
                            // We visited this VFSubgraph before, in such case check if op visited before had same
                            // tiling strategy as current one
                            VPUX_THROW_WHEN(itr->second.first != it->second,
                                            "Got mismatched tiling strategies for VFRegion: {0}",
                                            getOpHash(parentVerticalFusionOp));
                            continue;
                        }
                        parentVerticalFusionOp.getOperation()->setAttr(tilingStrategy, manualAttribute);
                    } else {
                        Logger::global().warning("Overwrite manual strategy {0} for op. opName {1}, opLoc {2}",
                                                 manualAttribute, opName, op.second->getLoc());
                        op.second->setAttr(tilingStrategy, manualAttribute);
                    }
                } else {
                    auto attrType = it->first.str();
                    VPUX_THROW_WHEN(!isAllowedAttr(attrType), "Unsupported Attribute '{0}'", it->first.str());
                }
            } else {
                auto attrType = it->first.str();
                if (op.second->hasAttr(attrType) && !isAllowedAttr(attrType)) {
                    // currently no default value, to disable multiclustering remove the attribute
                    op.second->removeAttr(attrType);
                }
            }
        }
    }
}

void updateAttributeValue(llvm::json::Value& json, const std::string& opName, StringRef attribute,
                          llvm::json::Value&& newValue) {
    auto jsonAsObject = json.getAsObject();
    if (jsonAsObject == nullptr) {
        return;
    }

    if (jsonAsObject->find(opName) == jsonAsObject->end()) {
        return;
    }

    auto jsonOpsToAttributes = jsonAsObject->operator[](opName).getAsObject();
    if (jsonOpsToAttributes == nullptr) {
        return;
    }

    auto jsonAttrsToLayerAttribute = jsonOpsToAttributes;
    if (jsonAttrsToLayerAttribute->find(attribute.str()) == jsonAttrsToLayerAttribute->end()) {
        return;
    }

    (*jsonAttrsToLayerAttribute)[attribute.str()] = std::move(newValue);
}

// find tilingStrategy JSON value in JSON file via key 'opName' and update this JSON value according to operation's
// tilingStrategy attribute
void updateTilingStrategyInJSONForOperations(llvm::json::Value& json,
                                             llvm::MapVector<mlir::Location, mlir::Operation*>& operations) {
    for (auto& op : operations) {
        auto opName = vpux::stringifyPrimaryLocation(op.first);
        auto parentClusterOp = op.second->getParentOfType<VPU::NCEClusterTilingOp>();

        // If opName is found, retrieve previous attribute from json
        auto prevAttributeValue = getPreviousAttributeValue(json, opName, tilingStrategy);
        if (!prevAttributeValue.has_value()) {
            continue;
        }

        llvm::json::Value currAttributeValue(defaultNoValue);
        if (op.second->hasAttr(tilingStrategy)) {
            // Get value present in IR
            currAttributeValue = convertAttrToJSON(op.second->getAttr(tilingStrategy));
        } else if (parentClusterOp != nullptr && parentClusterOp->hasAttr(tilingStrategy)) {
            // Get value from parentClusterOp
            currAttributeValue = convertAttrToJSON(parentClusterOp->getAttr(tilingStrategy));
        }

        // Update tiling strategy attribute in json
        if (prevAttributeValue.value() != currAttributeValue) {
            Logger::global().warning("Update tiling strategy in JSON. opName {0}, opLoc {1}, from {2} to {3}", opName,
                                     op.second->getLoc(), prevAttributeValue.value(), currAttributeValue);
            updateAttributeValue(json, opName, tilingStrategy, std::move(currAttributeValue));
        }
    }
}

void saveMCSideLoadStrategyToFile(mlir::func::FuncOp func, StringRef strategyJsonPath,
                                  const mlir::DenseMap<mlir::Operation*, size_t>& opToHash, StringRef modelHash) {
    constexpr StringLiteral MODEL_HASH_KEY = "ModelHash";
    constexpr StringLiteral OP_HASH_KEY = "Op";

    llvm::json::Object model{};
    model[MODEL_HASH_KEY.str()] = modelHash.str();
    llvm::json::Object opsToStrategies{};
    func->walk([&](VPU::LayerOpInterface op) {
        auto isNCEOp = mlir::isa<VPU::NCEOpInterface>(op.getOperation());
        auto isSWOp = mlir::isa<VPU::SWOpInterface>(op.getOperation());
        if (!isNCEOp && !isSWOp) {
            return;
        }
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op.getOperation());
        if (clusteredOp == nullptr) {
            return;
        }
        VPUX_THROW_WHEN(opToHash.find(op.getOperation()) == opToHash.end(), "Can not find hash at '{0}'", op.getLoc());

        auto strategyValue = clusteredOp.getMultiClusterStrategy().has_value()
                                     ? convertAttrToJSON(op->getAttr(vpux::multiClusterStrategy))
                                     : llvm::json::Value(defaultNoValue);

        llvm::json::Object layerAttributes{};
        layerAttributes[vpux::multiClusterStrategy.str()] = std::move(strategyValue);
        auto layerHashStr = std::to_string(opToHash.at(op.getOperation()));
        opsToStrategies[std::move(layerHashStr)] = llvm::json::Value(std::move(layerAttributes));
    });
    model[OP_HASH_KEY.str()] = std::move(opsToStrategies);
    auto json = llvm::json::Value(std::move(model));
    writeManualStrategyJSON(strategyJsonPath, json);
}

}  // namespace VPU
}  // namespace vpux

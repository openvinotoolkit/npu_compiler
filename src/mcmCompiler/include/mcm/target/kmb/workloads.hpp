//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
#ifndef MV_WORKLOADS
#define MV_WORKLOADS

#include <string>
#include <array>
#include <functional>
#include "include/mcm/computation/model/model_element.hpp"
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/base/exception/op_error.hpp"
#include "include/mcm/base/exception/index_error.hpp"
#include "include/mcm/tensor/shape.hpp"
#include "include/mcm/target/kmb/rectangle.hpp"
#include "include/mcm/target/kmb/workload_struct.hpp"
#include <climits>
#include <math.h>
#include <cmath>
#include <vector>
#include <numeric>

namespace mv
{
    /**
    * @brief Cost Function types to be used when evaluating execution cycles of a workload
    */
    enum class CostFunctions
    {
        Balanced,
        CriticalPath,
        Greedy,
        MinMaxWorkloads
    };

    /* The compiler generates a lattic structure of the tensor shape with the nodes numbered in this order
     * Example for tensor size 16x16
     *
     *         axis numbering
     *
     *         0    4     8    12
     *
     *    0    0----2-----4----6 //Even numbers
     *         |    |     |    |
     *    4    1----3-----5----7 //Odd numbers
     *         |    |     |    |
     *    8    10---11----12---13 // Incrementing numbers
     *         |    |     |    |
     *    12   15---16----17---18
     */

    struct point
    {
        int16_t x = 0;
        int16_t y = 0;
    };

    enum class WorkloadSplitMode { HW=1, HC, WC , NC, C, H};

    class Workloads : public LogSender
    {

        std::vector<Workload> workloads_;
        std::string layerName_;
        mv::Shape tensorShape_;
        bool isSparse_;
        std::vector<float> executionCycles_; //Min & Max execution cycles
        float meanExecutionCycles_ = 0;

        float critical_workload_ = 0;
        float workload_sum_ = 0;
        float min_range_ = 0;
        float max_range_ = 0;

    public:
        Workloads(const std::string& name, const mv::Shape& tensorShape);
        Workloads(const std::string& name, const mv::Shape& tensorShape, const bool& isSparse);
        ~Workloads();

        int partitionTensorWithRectangleHeuristic(const mv::DPUModeList& modes,
                                                            size_t        nWorkloads,
                                                            bool         split_over_h,
                                                            bool         split_over_w,
                                                            bool         split_symmetric,
                                                  const mv::WorkloadSplitMode& split_mode,
                                                  const mv::pass::PassEntry& pass);

        int partitionTensorWithZsplit(const mv::DPUModeList& modes, size_t nWorkloads, const mv::pass::PassEntry& pass, std::vector<std::size_t>& validZTiles);

        void populateWorkloadsFromPartitions(size_t nWorkloads,
                                            const mv::pass::PassEntry& pass,
                                            mv::DPUMode& mpeMode);

        std::vector<mv::Workload> polygonWorkloadSplit(const mv::pass::PassEntry& pass,
                                                        mv::Workload& workload,
                                                        std::vector<mv::Workload>& workloads,
                                                        mv::DPUMode& mpeMode);

        std::vector<mv::Workload> workloadSplitHelper(const mv::pass::PassEntry& pass,
                                                        mv::Workload& workload,
                                                        std::pair<std::pair<int16_t, int16_t>,bool>& interesting_point,
                                                        mv::DPUMode& mpeMode);

        void add_xy_offset(std::vector<std::size_t>& offset);
        void apply_z_offset(std::vector<std::size_t>& offset);
        void populateClusterID(int clusterID);

        std::size_t nWorkloads() const;
        void addWorkload(mv::Workload workload);
        const std::vector<mv::Workload>& getWorkloads() const;
        static const std::vector<int> getWorkloadSplitPool(const Tensor& tensor, int nDPUxCluster, mv::DPUModeList dpuModeList, int maxSplits, std::vector<std::size_t> valid_ztiling={16});

        static void generateExecutionCycles(std::vector<mv::Workloads>& workloadsVector, int nDPUxCluster, CostFunctions costFunction, float pixelCost, int workloadCost);
        std::vector<float> getExecutionCycles() const;
        float getMeanExecutionCycles() const;
        void setExecutionCycles(std::vector<float> val);
        static float greedyTaskAssignment(int nProcessors, std::vector<float>& workloadCosts);

        bool validateWorkloads(const mv::Shape& shape);

        static mv::CostFunctions getCostFunction(mv::Element& passDesc, const mv::pass::PassEntry& pass);

        double getAllWorkloadsVolume() const;
        bool noOverlap() const;
        mv::Shape getShapefromMinMax() const;

        Workload& operator[](int nworkload);
        bool operator < (const mv::Workloads& other) const;

        const Workload& operator[](int nworkload) const;
        std::string getLogID() const override;
        std::string toString() const;
        std::string toLongString() const;
        const std::vector<mv::Workload>& overlap_and_clip(std::array <unsigned short, 4>& padding, const Shape& tensorShape);
    };
}

#endif


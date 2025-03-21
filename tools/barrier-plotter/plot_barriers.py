#!/usr/bin/env python3
#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
import subprocess
import sys, os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('prefix', nargs='?', default="", help='prefix')
parser.add_argument("--max-slot-count", dest='max_slot_count', type=int, default=64, help="Slot count above which barriers have red background")
parser.add_argument("-u", dest="updateBarriersFile", type=str, default='.taskUpdateBarriers', help="Update barriers file")
parser.add_argument("-w", dest="waitBarriersFile", type=str, default='.taskWaitBarriers', help="Wait barriers file")
parser.add_argument("-f", dest="taskQueueTypeMapFile", type=str, default='.taskQueueTypeMap', help="Task queue type map file")
parser.add_argument("-s", dest="slotCountFile", type=str, default='.slots', help="Slot counts file")
parser.add_argument("-o", dest="outfilePref", type=str, help="Output file prefix")
parser.add_argument("--fromTask", dest="fromTask", type=int, default=0, help="Read input files starting from this task")
parser.add_argument("--toTask", dest="toTask", type=int, default=-1, help="Read input files up to this task")
parser.add_argument("--lines", type=str, default="lines", help='''Connection line style. `lines` can be faster to export to pdf, 
                    but some connections may overlap.''', choices=['lines', 'splines'])
args = parser.parse_args()

def getFname(fname):
    if os.path.isfile(fname):
        return fname
    elif os.path.isfile(args.prefix+fname):
        return args.prefix+fname
    else:
        return args.outfilePref+fname

def load(fname, args):
    with open(fname, mode='r') as f:
        depsMap=[]
        if args.toTask == -1:
            for l in f.readlines():
                a=np.array(l.rstrip().split(' '))
                depsMap.append(np.array(a[1:], dtype=int))
        else:
            for i, l in enumerate(f.readlines()):
                if i >= args.fromTask and i <= args.toTask:
                    a=np.array(l.rstrip().split(' '))
                    depsMap.append(np.array(a[1:], dtype=int))

    offset = args.fromTask
    return depsMap, offset

def loadFIFO(fname, args):
    with open(fname, mode='r') as f:
        depsMap=[]
        for l in f.readlines():
            _, fvals = l.rstrip().split(':')
            a=np.array(fvals.split(' '))
            a=np.array(a[1:], dtype=int)
            a=a[a>=args.fromTask]
            if args.toTask != -1:
                a=a[a<=args.toTask]
            depsMap.append(np.array(a, dtype=int))

    offset = args.fromTask
    return depsMap, offset

def printDAG(updateBarriers, waitBarriers, offset, taskQueueTypeMap, slotCount):
    barriers = set()
    for taskId, bars in enumerate(updateBarriers + waitBarriers):
        for b in set(bars):
            barriers.add(b)

    #
    # create barrier producer and consumer maps 
    #
    barProducers = [ list() for _ in range(max(barriers) + 1)]
    barConsumers = [ list() for _ in range(max(barriers) + 1)]

    for taskId, bars in enumerate(updateBarriers, offset):
        for b in set(bars):
            barProducers[b].append(taskId)

    for taskId, bars in enumerate(waitBarriers, offset):
        for b in set(bars):
            barConsumers[b].append(taskId)

    with open(outputFile+'.dot', 'w') as f:
        f.write("digraph schedule {\n")
        if args.lines == 'lines':
            f.write("graph [splines=line];\n")
        elif args.lines == 'splines':
            f.write("graph [splines=true overlap=false];\n")
        else:
            pass
        
        #
        # print dot graph
        # 
        for b in barriers:
            prodCount = sum(slotCount[x-offset] for x in set(barProducers[b]))
            consCount = sum(slotCount[x-offset] for x in set(barConsumers[b]))
            slotsSum = prodCount + consCount
            lab = 'b'+str(b)+ "|p: %i |c: %i |sum: %i" % (prodCount, consCount, slotsSum )
            if slotsSum >= args.max_slot_count:
                f.write('b'+str(b) + ' [shape=box, style=filled, color=red, label="' + lab + '"];\n')
            else:
                f.write('b'+str(b) + ' [shape=box, color=red, label="' + lab + '"];\n')

        for taskId, bars in enumerate(updateBarriers, offset):
            for bar in bars:
                f.write(str(taskId) + ' -> b'+str(bar) + ';\n')

        for taskId, bars in enumerate(waitBarriers, offset):
            for bar in bars:
                f.write('b'+str(bar) + ' -> ' + str(taskId)+';\n')

        #
        # print FIFO connections
        # 
        queueColors = ['red', 'blue', 'green','cyan', 'magenta', 'yellow', 'grey', 'brown', 'indigo', 'lavender', 'purple', 'turquoise', 'slateblue', 'darkslateblue', 'skyblue', 'olivedrab']
        for i,queue in enumerate(taskQueueTypeMap):
            s=''
            if len(queue) > 0:
                for task in queue[:-1]:
                    s=s + str(task) + "->" 
                s=s + str(queue[-1])

                f.write(s + ' [color='+queueColors[i]+'];\n')

        f.write("}")

    cmd = 'dot -Tpdf %s.dot -O' % outputFile
    print("Exporting to pdf")
    print(cmd)
    subprocess.run(cmd.split())

#
# configure input files
#
updateBarriersFile = getFname(args.updateBarriersFile)
waitBarriersFile = getFname(args.waitBarriersFile)
slotCountFile = getFname(args.slotCountFile)
outputFile = args.outfilePref

#
# load graph data
#
updateBarriers, _ = load(updateBarriersFile, args) 
waitBarriers, offset = load(waitBarriersFile, args)

if args.toTask > 0:
    slotCount=np.loadtxt(slotCountFile)[:,1][args.fromTask:args.toTask+1]
else:
    slotCount=np.loadtxt(slotCountFile)[:,1][args.fromTask:]

taskQueueTypeMap = []
if args.taskQueueTypeMapFile != "":
    taskQueueTypeMap, _ = loadFIFO(getFname(args.taskQueueTypeMapFile), args)

#
# generate dot and export to pdf
# 
printDAG(updateBarriers, waitBarriers, offset, taskQueueTypeMap, slotCount)

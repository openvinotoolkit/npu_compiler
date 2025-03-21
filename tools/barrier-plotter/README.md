Barrier plotter is a visualization tool to create `dot` graphs from input data representing state of the 
control graph (containing tasks, barriers, FIFOs and their dependencies) at chosen stage of compilation.
Example usage:

```python
plot_barriers.py test -o test.config
```

For large models, exporting to PDF can take a significant amount of time. In such cases it is better to generate
the dot file only for some range of tasks. This is done by providing range arguments

```python
plot_barriers.py test -o test.config --from 1 --to 8
```
However, currently this will affect barrier slot counts, hence appropriate margin should be assumed around 
tasks of interest in the model.

It is possible to generate `dot` without providing FIFO information:
```python
plot_barriers.py -u test.taskUpdateBarriers -w test.taskWaitBarriers -s test.slots -f '' -o test.noFifo
```
For the case of graph not fully controlled by barriers, different parts of the graph will be disconnected.

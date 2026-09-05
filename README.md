# Black Hole Sun

`black-hole-sun` is an extension of the [Jungle](https://github.com/nicksenger/Jungle) "Workflow-as-Type" (WaT) orchestration system which is intended to be used for **machine learning**.

It enables creation of distributed pipelines for processing any data expressable as statically typed tensors (via [Glowstick](https://github.com/nicksenger/Glowstick)). Use-cases include agent development, model training, etc.

`black-hole-sun` also includes a GUI monitoring tool with an optional piano:

https://github.com/user-attachments/assets/ef463b7b-c1fd-455a-94fe-c712913842ab

A `BlackHole::Sun` graph is a hierarchical, static-topology DAG where each node performs some operation over an input tensor of known shape to produce an output tensor of known shape. This could be an individual matmul, prompting a gigantic LLM, making an API call, or even just evaluating some logical expession.

The role of `black-hole-sun` is to:
1. ensure, to the extent possible, that any defined networks and traversals thereof are valid at compile time.
2. orchestrate the movement of data efficiently and reliably through these networks.

`black-hole-sun` is an art project and research tool. Documentation is sparse, and the interface should be considered highly unstable. There are some examples of training a ResNet with it using a variety of methods in `_black_hole_toy`.


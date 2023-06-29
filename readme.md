### TODO

- Fix FIXMEs and TODOs scattered throughout the code.
  - [x] Split data processing from `TypedGraphDataset` base class.
  - [x] Integrate new Dataset base and bubble classes.
    - [x] Update `BubbleSequenceDataset`.
  - [x] Implement a `FlagDataset` that uses the new interfaces.
  - [x] Implement different index types.
  - [x] Incorporate global state.
    - [x] Add the option to provide a base `TypedGraphLayout` to a `TypedGraph`
      object. Nodes and edges can then be added as if the base layout is part
      of the current object.
    - [x] Add `merge` and `extract` operations to merge a graph with another
      graph that conforms to the provided base layout.
    - [x] Remove internal references to global state, internal constructs
      should operate on a single graph.
    - [x] Provide suitable public interfaces that accept a `TypedGraph` derived
      data item and global state. These interfaces should handle merging and
      splitting of these graphs before and after handling by internal
      machinery, respectively.
  - [x] Fix `Bubble` and `Flag` models to use the global passes.
  - [x] Make sure `source` and `target` are correctly handled when evaluating
    indices.
  - [x] Expand `NodeSets` (and `EdgeSets`?) to be able to contain more than one
    attribute.
    - [x] Update `NodeSet`, and `NodeSetSummary`.
    - [x] Implement handling of the base layout in `NodeSet`
    - [x] Implement updated merging and extracting of base- and supergraphs.
    - [x] Update `EPDConfig`, and `GraphPassConfig`.
    - [x] Update inner workings of node related functions in `EncodeProcessDecode`
      and `GraphPass`.
    - [x] Build example of more complex transfer function, e.g. equivariant mesh
      transfer.
- [ ] Add instructions on how to run the models.
- [ ] Add missing training routines, e.g. `TBPTT`.
- [ ] Add missing slurm job files.
- [ ] Document the framework.

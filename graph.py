"""
A small, dependency-free orchestration engine deliberately modeled on the
LangGraph programming model: named nodes (functions of `state -> state`),
directed edges, conditional edges (a router function returning the name of
the next node), and a designated END sentinel.

Why not just import `langgraph` directly? Two reasons:
1. Zero extra dependency / zero network / zero version-skew risk for a
   demo project that should run out of the box.
2. The state object here (PipelineState) is a plain dataclass rather than a
   TypedDict-of-reducers, which keeps the example easy to read end-to-end.

The API is intentionally close to LangGraph's `StateGraph` so that swapping
this engine for the real `langgraph.graph.StateGraph` is a small, mechanical
change (see README.md "Swapping in real LangGraph").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

State = TypeVar("State")
NodeFn = Callable[[State], State]
RouterFn = Callable[[State], str]

END = "__end__"


class GraphError(RuntimeError):
    pass


@dataclass
class _ConditionalEdge:
    router: RouterFn
    path_map: dict[str, str]


class StateGraph(Generic[State]):
    def __init__(self):
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._conditional_edges: dict[str, _ConditionalEdge] = {}
        self._entry_point: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph[State]":
        if name == END:
            raise GraphError(f"'{END}' is reserved and cannot be used as a node name.")
        self._nodes[name] = fn
        return self

    def set_entry_point(self, name: str) -> "StateGraph[State]":
        self._entry_point = name
        return self

    def add_edge(self, from_node: str, to_node: str) -> "StateGraph[State]":
        self._edges[from_node] = to_node
        return self

    def add_conditional_edges(
        self, from_node: str, router: RouterFn, path_map: dict[str, str]
    ) -> "StateGraph[State]":
        self._conditional_edges[from_node] = _ConditionalEdge(router=router, path_map=path_map)
        return self

    def compile(self, max_steps: int = 200) -> "CompiledGraph[State]":
        if self._entry_point is None:
            raise GraphError("No entry point set. Call set_entry_point() before compile().")
        for name in list(self._edges.values()) + [self._entry_point]:
            if name != END and name not in self._nodes:
                raise GraphError(f"Edge references undefined node '{name}'.")
        return CompiledGraph(self, max_steps=max_steps)


class CompiledGraph(Generic[State]):
    def __init__(self, graph: StateGraph[State], max_steps: int = 200):
        self._graph = graph
        self._max_steps = max_steps
        self.trace: list[str] = []

    def invoke(self, initial_state: State) -> State:
        state = initial_state
        current = self._graph._entry_point
        steps = 0
        self.trace = []
        while current != END:
            steps += 1
            if steps > self._max_steps:
                raise GraphError(
                    f"Graph execution exceeded max_steps={self._max_steps}; "
                    "likely an unintended infinite loop in conditional edges."
                )
            if current not in self._graph._nodes:
                raise GraphError(f"No such node: '{current}'")
            self.trace.append(current)
            fn = self._graph._nodes[current]
            state = fn(state)

            if current in self._graph._conditional_edges:
                cond = self._graph._conditional_edges[current]
                decision = cond.router(state)
                if decision not in cond.path_map:
                    raise GraphError(
                        f"Router at node '{current}' returned '{decision}', which is "
                        f"not in its path_map {list(cond.path_map.keys())}."
                    )
                current = cond.path_map[decision]
            elif current in self._graph._edges:
                current = self._graph._edges[current]
            else:
                current = END
        self.trace.append(END)
        return state

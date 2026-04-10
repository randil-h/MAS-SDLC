"""
LangGraph pipeline definition for the MAS SDLC system.

Builds a deterministic, sequential StateGraph that routes the shared SDLCState
through four agent nodes:
  requirements → code_generator → test_engineer → code_reviewer
"""

from langgraph.graph import END, START, StateGraph

from agents.code_generator_agent import code_generator_node
from agents.code_reviewer_agent import code_reviewer_node
from agents.requirements_agent import requirements_node
from agents.test_engineer_agent import test_engineer_node
from state import SDLCState


def build_graph() -> StateGraph:
    """
    Construct and compile the SDLC LangGraph pipeline.

    The graph is fully sequential with no conditional branching.  Each agent
    node receives the complete SDLCState, updates its designated field(s), and
    passes the enriched state to the next node.

    Pipeline
    --------
    START → requirements → code_generator → test_engineer → code_reviewer → END

    Returns
    -------
    StateGraph
        A compiled LangGraph graph ready to be invoked via ``graph.invoke(state)``.
    """
    graph = StateGraph(SDLCState)

    # Register nodes
    graph.add_node("requirements", requirements_node)
    graph.add_node("code_generator", code_generator_node)
    graph.add_node("test_engineer", test_engineer_node)
    graph.add_node("code_reviewer", code_reviewer_node)

    # Sequential edges
    graph.add_edge(START, "requirements")
    graph.add_edge("requirements", "code_generator")
    graph.add_edge("code_generator", "test_engineer")
    graph.add_edge("test_engineer", "code_reviewer")
    graph.add_edge("code_reviewer", END)

    return graph.compile()

"""Creates the workflow graph to run the Agents and tools."""

import functools
import json
import os
import sys

# Add parent directory to the system search path.
sys.path.append(os.getcwd())
print(sys.path)

import numpy as np
import ollama

from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod

from config import prompt_templates
from scisr_lm import scisr_memory
from scisr_lm import scisr_agents
from scisr_lm import agent_response_struct
from tools import statistics_compute_tools
from tools import prepare_final_answer_tools

from unit_tests import test_symbolic_expr_prompts


def create_and_rungraph(llm:str, global_system_prompt:str, prompt_tools:str, tools_dict:dict):
  """Creates the workflow for scisr language model."""
  # Step1: Create a workflow, this will hold all the 
  workflow = StateGraph(scisr_agents.State)

  # Step2: Create an entry for the workflow.
  node_agent_func = functools.partial(
    scisr_agents.node_agent, 
    llm=llm, global_system_prompt=global_system_prompt, 
    prompt_tools=prompt_tools, tools_dict=tools_dict)
  workflow.add_node(node="Agent", action=node_agent_func)
  workflow.set_entry_point(key="Agent")

  # Step3: Add other nodes in the workflow. These are the tools for the agent.
  for k in tools_dict.keys():
    workflow.add_node(node=k, action=scisr_agents.node_tool)

  # Step4: Add conditional edges from the Agent.
  workflow.add_conditional_edges(source="Agent", path=scisr_agents.conditional_edge)

  # Step5: Add Normal edges to Agent.
  for k in tools_dict.keys():
    if k != 'final_answer':
      workflow.add_edge(start_key=k, end_key="Agent")

  # Step6: End the graph by creating a edge between the final_answer node to the End node.
  workflow.add_edge(start_key="final_answer", end_key=END)

  g = workflow.compile()

  display(Image(
    g.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
    ))

  return g



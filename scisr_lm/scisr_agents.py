"""Creates the necessary nodes, state and edges."""
import os
import sys
import logging

from typing import TypedDict

import ollama

from scisr_lm import agent_response_struct
from scisr_lm import scisr_memory

AgentRes = agent_response_struct.AgentRes


class State(TypedDict):
  """State represents the information passed between nodes."""
  user_q: str  # Never take off from the original user query.
  chat_history: list  # Warmup chats between assistant and user.
  lst_res: list[AgentRes]  # List of responses from the previous agents and tools.
  output: dict  # Output from the current node.


######################################################
# Both Agent and Tool will have a node of their own. And for each node, we should write
# a function to decide on its action. And for each edge, we need to write a function 
# on the communication between the nodes.
# 
# The edges can be of two types: conditional edge and normal edge.
#   Conditional edge: This will help to decide on the next node from the current node.
#   Normal edge: Directly goes from one node to another.
######################################################

def node_agent(state: State, llm:str, global_system_prompt:str, prompt_tools:str, tools_dict:dict):
  """Runs the agent determined by this node."""
  logging.info('Running node agent with state: %s', state)
  agent_res = run_agent(
    llm=llm,
    global_system_prompt=global_system_prompt,
    prompt_tools=prompt_tools,
    tools_dict=tools_dict,
    user_q=state["user_q"],
    chat_history=state["chat_history"],
    lst_res=state["lst_res"],
    )
  logging.debug('agent_res: %s', agent_res)
  return {"lst_res": [agent_res]}


def node_tool(state):
  """Ensure that the tool's response adheres to AgentRes standard."""
  logging.info('Running node tool.')
  res = state["lst_res"][-1]

  logging.debug('Current running tool: %s', res)
  agent_res = AgentRes(
      tool_name=res.tool_name,
      tool_input=res.tool_input,
      tool_output=res.tool_output,
    )
  return {"output":agent_res} if res.tool_name == "final_answer" else {"lst_res":[agent_res]}


def conditional_edge(state):
  """Directs the LLMs to produce final answer or considering using another tool."""
  last_res = state["lst_res"][-1]
  next_node = last_res.tool_name if isinstance(state["lst_res"], list) else "final_answer"
  logging.debug('Next node selected is : %s', next_node)
  return next_node


def run_agent(llm:str, global_system_prompt:str, prompt_tools:str, tools_dict:dict, user_q:str, chat_history:list[dict], lst_res:list[AgentRes]) -> AgentRes:
  """Indicates a single run of the agent.

  Args:
    llm: Choices of available llms.
    global_system_prompt: Global system prompt to prepare the LLM.
    prompt_tools: Prompt to use the tools.
    tools_dict: A dictionary object with tool name and actual tool.
    user_q: Original user query for which the answers are expected.
    chat_history: Initial prompt for the system.
    lst_res: List of responses from the Agents used previously.

  Return:
    An AgentRes object containing the results from using the tool.
  """
  # For each agent, we create the memory that includes past agent responses captured in the
  # lst_res.
  memory = scisr_memory.save_memory(lst_res=lst_res, user_q=user_q)

  # Check the past tool usage and ensure that each tool should be used only once.
  if memory:
    tools_used = [res.tool_name for res in lst_res]
    if len(tools_used) >= len(tools_dict.keys()):
      memory[-1]["content"] = 'You must now use the  "final_answer" tool.'

  # Preparing the inputs for the agents to now use the past conversations to take the next step.
  messages = [
    {"role": "system", "content": global_system_prompt+"\n"+prompt_tools},
    *chat_history,
    {"role": "user", "content":user_q},
    *memory
  ]

  logging.info("\n Messages during agent run: \n%s\nEnd of Agent run message.\n", messages)
  llm_res = ollama.chat(model=llm, messages=messages, format="json")
  return AgentRes.from_llm(llm_res)

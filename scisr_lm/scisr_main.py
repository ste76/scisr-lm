"""Main run script to run LLM, Agents and tools.

Following functionalities are implemented in this main script.
  - Checks if a LLM model is served via OLLAMA. Backup would be to use hosted LLMs such as 
      GPT4o-mini
  - Runs streamlit to create the UI screen that includes Chat
  - Background or preparatory steps:
    - Initiation of AI Agents
    - Initiation of the Tools: Statistical and Neural tools.
  - Chat and related Steps:
    - Vector database for memory
    - Chat session memory (long term and short term memory)
"""

import argparse
import inspect
import logging
import json
import os
import random
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
from scisr_lm import scisr_graph
from scisr_lm import agent_response_struct
from tools import statistics_compute_tools
from tools import prepare_final_answer_tools

from unit_tests import test_symbolic_expr_prompts


def run_main(params):
  """Main script to organize key functions of Scisr-LM."""
  lang_model = params.get("lang_model", "llama3.1")

  # Get a simple user query.
  user_q = test_symbolic_expr_prompts.prompt1_dict["prompt"]
  logging.info('The input prompt is: %s', user_q)

  # Creation of input assistant prompts and tools.
  prompt = prompt_templates.global_system_prompt

  # Tools that will be available for the LLMs to use.
  tools_dict = {
    'compute_outliers': statistics_compute_tools.compute_outliers_tool,
    'compute_iqr': statistics_compute_tools.compute_iqr_tool,
    'compute_mean': statistics_compute_tools.compute_mean_tool,
    'compute_kurtosis': statistics_compute_tools.compute_kurtosis_tool,
    'compute_skewness': statistics_compute_tools.compute_skewness_tool,
    'compute_mode': statistics_compute_tools.compute_mode_tool,
    'compute_median': statistics_compute_tools.compute_median_tool,
    'compute_variance': statistics_compute_tools.compute_variance_tool,
    'final_answer': prepare_final_answer_tools.final_answer_tool,
  }

  tools_dict = {
    'compute_variance': statistics_compute_tools.compute_variance_tool,
    'final_answer': prepare_final_answer_tools.final_answer_tool,
  }

  tools_str = "\n".join([str(n+1)+". `"+str(v.name)+"`: "+str(v.description) for n,v in enumerate(tools_dict.values())])
  prompt_tools = f"You can use the following tools:\n{tools_str}"

  logging.info("You can use the following tools: \n %s", prompt_tools)

  # Step1: Create a workflow graph.
  workflow_graph = scisr_graph.create_and_rungraph(
    llm=lang_model, global_system_prompt=prompt, prompt_tools=prompt_tools, tools_dict=tools_dict)

  # Step2: Initialize the inputs for the graph.
  initial_state = {'user_q':user_q,
                   'chat_history':prompt_templates.chat_history, 
                   'lst_res':[],
                   'output':{}}

  # Step3: Invoke and run the graph.
  out = workflow_graph.invoke(input=initial_state)
  agent_out = out['output'].tool_output

  logging.info('Final response from the LMs: \n\n%s\n\nEnd of final Response.', agent_out)



if __name__ == "__main__":
  FORMAT = '[%(asctime)s-%(filename)s %(lineno)s]: %(funcName)s-%(message)s'
  logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('-lm', '--lang_model', 
    dest='lang_model',
    default='llama3.1',
    type=str, help='Path to the output directory to store the dataset.')

  args = parser.parse_args()
  params = vars(args)

  logging.info('Input Arguments: %s\n', json.dumps(params, indent=2))

  run_main(params)

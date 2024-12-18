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

import constants
import symbolic_expr_priors

from config import prompt_templates
from unit_tests import test_symbolic_expr_prompts


def run_main(params):
  """Main script to organize key functions of Scisr-LM."""
  lang_model = params.get("lang_model", "llama3.3")


if __name__ == "__main__":
  FORMAT = '[%(asctime)s-%(filename)s %(lineno)s]: %(funcName)s-%(message)s'
  logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('-lm', '--lang_model', 
    dest='lang_model',
    default='.',
    type=str, help='Path to the output directory to store the dataset.')

  args = parser.parse_args()
  params = vars(args)

  logging.info('Input Arguments: %s\n', json.dumps(params, indent=2))

  run_main(params)

"""Creates new prompts to query LLM, prepare training data and evaluate performance.

  A prompt will contain the following,
    - base prompt: This is a combination of system prompt with user prompt. There will be
        place_holders for input data and user-query.
    - Input/output data pair: A sequence of [(x1, y1), (x2, y2),....(x100, y100)] generated
        from a sample of known guassian prompt
    - User_query: the actual that we want the llm to answer based on the system message and
        input data.

  A Hint will be generated based on the input range and the functions that was used to 
    create the data pairs. The hints can be supplied to the LLM or provided to the humans
    creating the training dataset.

Output:
  Example1:
    prompt : This will be the original prompt: base+data_pairs_user_query
    Hint: This will be the symbol category that would have been used to generate the data_pairs.
    Human_hint: Contains actual function with available symbols.

Usage:
  python symbolic_priors/simulate_prompts.py  --output_dir="./datasets" --variable_cnt=1 --num_prompts=1000000 --user_query_file='./config/user_queries.txt'
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

import constants
import symbolic_expr_priors

from config import prompt_templates

# Definition of the constants.
NUM_DATA_PAIRS_ = 100


def generate_prompts(params):
  """Generates prompts by simulating the symbolic priors expression."""
  output_dir = params.get('output_dir', './dataset')
  variable_cnt = params.get('variable_cnt', 1)
  num_prompts = params.get('num_prompts', 100)
  user_query_file = params.get('user_query_file', './config/user_queries.txt')

  # Step1: Identify the expressions corresponding to the variable_cnt input
  #  parameters.
  if variable_cnt == 1:
    symbolic_expr_list = constants.SYMBOLIC_EXPR_PRIORS_ONE_VARIABLE
  elif variable_cnt == 2:
    symbolic_expr_list = constants.SYMBOLIC_EXPR_PRIORS_TWO_VARIABLE
  else:
    raise NotImplementedError

  # Step2: Collecting all user queries:
  all_user_queries = []
  with open(user_query_file, 'r') as f:
    for line in f.readlines():
      all_user_queries.append(line)

  # Preparing output write directory.
  try:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  except Exception as e:
    logging.error(f'dataset directory {output_dir} creation failed.')

  write_output = os.path.join(output_dir, f'symbolic_expr_prompts_{variable_cnt}.txt')

  # Step3: Sample available symbolic expression and preparing prompts for LLMs.
  all_prompts = []
  for i in range(num_prompts):
    example = {}
    if i%100 == 0:
      logging.info(f'Generated {i} prompts.')

    # Step4: Select a sample expressions.
    expr_index = random.choice(range(len(symbolic_expr_list)))
    symbolic_expr_dict = symbolic_expr_list[expr_index]

    x_range = symbolic_expr_dict['x']
    classes = list(symbolic_expr_dict['classes'])
    hint = symbolic_expr_dict['hint']

    symb_name_dict = symbolic_expr_priors.ALL_SYMBOLIC_PRIMITIVE_DICT
    symbol_names = [symb_name_dict[sym] for sym in classes]
    symbol_names_str = ', '.join(symbol_names)

    symbolic_expr_func = symbolic_expr_dict['function']
    key = str(symbolic_expr_func)+'_'+str(i)

    # Step4: Simulate the symbolic expression function based on the input range.
    data_pairs = []
    for j in range(NUM_DATA_PAIRS_):
      x = random.uniform(*x_range)
      y = symbolic_expr_func(x)
      data_pairs.append((x, y))

    # Sample a user query from the list of user queries.
    query_ind = random.choice(range(len(all_user_queries))) 
    user_query = all_user_queries[query_ind]

    # Create the human hint.
    func_def = inspect.getsource(symbolic_expr_func)
    human_hint = 'The actual function that were used is: \n\n{0} \n\n and uses the symbols: \n{1}'

    # Step5: Write TFRecord dataset in the output_dir
    example['key'] = key
    example['prompt'] = prompt_templates.system_prompt.format(str(data_pairs), user_query)
    example['hint'] = f'Is it possible that the input/output pairs is generated by {hint}'
    example['human_hint'] = human_hint.format(func_def, symbol_names_str)
    example['user_query'] = user_query
    all_prompts.append(example)

  # Step6: Write the created prompts.
  with open(write_output, 'w') as f:
    json.dump(all_prompts, f, indent=2)


if __name__ == "__main__":
  FORMAT = '[%(asctime)s-%(filename)s %(lineno)s]: %(funcName)s-%(message)s'
  logging.basicConfig(format=FORMAT, stream=sys.stdout, level=logging.INFO)

  parser = argparse.ArgumentParser()
  parser.add_argument('-od', '--output_dir', 
    dest='output_dir',
    default='../datasets',
    type=str, help='Path to the output directory to store the dataset.')
  parser.add_argument('-vc', '--variable_cnt',
    dest='variable_cnt',
    default=1,
    type=int, help='Expressions with variable_cnt parameters to simulate.')
  parser.add_argument('-np', '--num_prompts',
    dest='num_prompts',
    default=10,
    type=int, help='Total number of prompts that needs to be generated.')
  parser.add_argument('-qf', '--user_query_file',
    dest='user_query_file',
    default='./config/user_queries.txt',
    type=str, help='List of the possible user queries that can be asked about the data_pairs')

  args = parser.parse_args()
  params = vars(args)

  logging.info('Input Arguments: %s\n', json.dumps(params, indent=2))

  generate_prompts(params)

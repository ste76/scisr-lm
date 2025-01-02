"""Tools to help prepare the final answers."""
import os
import sys

from langchain_core.tools import tool

@tool("final_answer")
def final_answer_tool(text:str) -> str:
  """Returns a natural language response to the user by passing the input `text`. 
  You should provide as much context as possible and specify the source of the information.
  """
  return text
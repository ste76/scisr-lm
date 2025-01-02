"""To prevent LLMs from hallucinating, we need to add some structure to the response.
"""
from pydantic import BaseModel
import json
import logging

from exceptions.symbolic_exceptions import ScisrLMException

class AgentRes(BaseModel):
  """Forcing a structure to be followed by the LLMs.

  Each tool will have the following:
   tool_name: Name of the tool used.
   tool_input: Input parameters sent to the tool.
   tool_output: Response obtained from the tool.
  """
  tool_name: str  # Name of the tool: compute_outliers, compute_mean etc.
  tool_input: dict  # {'q': '''What is the statistical relation between input and output.'''}
  tool_output: str | None = None

  @classmethod
  def from_llm(cls, res:dict, tool_output: str=None):  # Returns the class itself.
    try:
      out = json.loads(res["message"]["content"])
      logging.info('\nResponse from tool:\n%s\n\n', res)
      logging.info('\nOutput from tool:\n%s\n\n', out)
      return cls(tool_name=out["name"], tool_input=out["parameters"], tool_output=tool_output)
    except Exception as e:
      msg = f"Error from Ollama: \n{res}\n"
      logging.error(msg)
      raise ScisrLMException(msg)
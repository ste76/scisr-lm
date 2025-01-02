"""Creates a memory system for the LLMs with Agent.

Content of the memory will have the following structure.
[
  {
   {"role": "assistant", "content": '{"name": "final_answer", 
                                      "parameters": {"text":"How can I assist you today?"}}'},
    {"role": "user", "content": None}
  }
]

"""
from typing import List

from scisr_lm.agent_response_struct import AgentRes

def save_memory(lst_res: List[AgentRes], user_q: str) -> List:
  """Saves the results from the previous transactions of LMs.

  Args:
    lst_res: List of the responses from AI Agents.
    user_q: Original input user query.

  Response:
    List of intermediate agentic responses followed by user query as a reminder.
  """
  memory = []  # Creates a fresh memory as the Previous agentic responses are captured.

  for agent_res in [res for res in lst_res if res.tool_output is not None]:
    memory.extend([
      # Intermediate query of the agent.
      {'role': 'assistant', 'content': json.dumps({"name": res.tool_name, "parameters": res.tool_input})},
      # Intermediate response of the agent.
      {'role': 'user', 'content': res.tool_output}
      ])

  # We need to add original query to avoid loosing query information.
  if memory:
    memory += [{
      'role': 'user', 'conent': (f'''
        This is just a reminder that my original query was `{user_q}`.
        Only answer to the original query, and nothing else, but use the information I gave you.
        Provide as much information as possible when you use the `final_answer` tool.''')
    }]

  return memory
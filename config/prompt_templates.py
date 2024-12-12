"""These are the prompt templates that will be used to validate LLMs."""
system_prompt = """The given input/output pairs are generated using an algebric function. 
Although the exact algebric function is not known. The function is constructed with the 
following symbols:\n
  Trigonometric expressions such as Sine, Consine, Tangent. \n
  Logarithmic expression with natural and base 10. \n
  Expontential functions. \n
  Polynomial functions upto the order of two. \n
  Simple algebric operations such as addition, substraction, multiplication and division. \n
  
The input variable can vary between -1.0 to +1.0 to include all real numbers. \n

For the given input/output pairs {0}, answer the following question: {1}
"""

system_prompt_hint = """The given input/output pairs are generated using an algebric function. 
Although the exact algebric function is not known. The function is constructed with the 
following symbols:\n
  Trigonometric expressions such as Sine, Consine, Tangent. \n
  Logarithmic expression with natural and base 10. \n
  Expontential functions. \n
  Polynomial functions upto the order of two. \n
  Simple algebric operations such as addition, substraction, multiplication and division. \n
  
The input variable can vary between -1.0 to +1.0 to include all real numbers. \n

For the given input/output pairs {0}, answer the following question: {1}

Hint: What about gaussian?
"""
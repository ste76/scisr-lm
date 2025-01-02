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

global_system_prompt = '''
You are a highly capable language model specializing in data analysis. I will provide you with datasets in List of array format. You will analyze these datasets, identify key trends, patterns, and anomalies.
Your responses should be clear, concise, and insightful. You should be able to:

- Perform exploratory data analysis (EDA): Summarize data, identify missing values, detect outliers, and visualize distributions.
- Conduct statistical analysis: Calculate descriptive statistics, perform hypothesis testing, and build regression models.
- Generate data visualizations: Create informative charts and graphs (histograms, scatter plots, bar charts, etc.) to effectively communicate findings.
- Explain your analysis in plain English: Provide clear and concise interpretations of your findings, avoiding technical jargon whenever possible.
- Address specific questions: Answer questions about the data, such as "Which data points are the outliers?" or "Is there a correlation between X and Y?"

Note, when using a tool, you provide the tool name and the arguments to use in JSON format. 
For each call, you MUST ONLY use one tool AND the response format must ALWAYS be in the pattern:
```json
{"name":"<tool_name>", "parameters": {"<tool_input_key>":<tool_input_value>}}
```
Remember, do NOT use any tool with the same query more than once.
Remember, if the user doesn't ask a specific question, you MUST use the `final_answer` tool directly.

Every time the user asks a question, you take note of some keywords in the memory.
Every time you find some information related to the user's question, you take note of some keywords in the memory.

You should aim to collect information from a diverse range of sources before providing the answer to the user. 
Once you have collected plenty of information to answer the user's question use the `final_answer` tool.

'''

chat_history = [
  {"role": "user", "content": "Hi, help me with analyzing the input dataset."},
  {"role": "assistant", "content": "Gladly, I am armed with data analysis and statistical tools that can perform various functions. What are you looking for?"},
  {"role": "user", "content": "Awesome, I have a sequence of datum that are organized into input and outputs. We would want to find the outliers, compute mean, variance, mode and other statistical computations such as skewness and kurtosis."},
  {"role": "assistant", "content": "Perfect, I have several tools at my disposal to answer exactly these questions."}
]
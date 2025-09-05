import os
from dotenv import load_dotenv
load_dotenv()  # Automatically looks for ".env"

api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = api_key


import pandas as pd
from docx import Document
import matplotlib.pyplot as plt


import json
import time
import traceback
from litellm import completion
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any
@dataclass
class Prompt:
    messages: List[Dict] = field(default_factory=list)
    tools: List[Dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # Fixing mutable default issue


def generate_response(prompt: Prompt) -> str:
    """Call LLM to get response"""

    messages = prompt.messages
    tools = prompt.tools

    result = None

    if not tools:
        response = completion(
            model="openai/gpt-4o",
            messages=messages,
            max_tokens=1024
        )
        result = response.choices[0].message.content
    else:
        response = completion(
            model="openai/gpt-4o",
            messages=messages,
            tools=tools,
            max_tokens=1024
        )

        if response.choices[0].message.tool_calls:
            tool = response.choices[0].message.tool_calls[0]
            result = {
                "tool": tool.function.name,
                "args": json.loads(tool.function.arguments),
            }
            result = json.dumps(result)
        else:
            result = response.choices[0].message.content


    return result



@dataclass(frozen=True)
class Goal:
    priority: int
    name: str
    description: str

goals = [
    Goal(
        priority=1,
        name="Business Analyst AI Agent",
        description="Act as a Business Analyst AI agent."
    ),
    Goal(
        priority=2,
        name="File Analysis",
        description="Read and analyze Excel (.xlsx), CSV, and Word (.doc/.docx) files."
    ),
    Goal(
        priority=3,
        name="Answer Questions",
        description="Answer user questions about the data in clear, professional business language."
    ),
    Goal(
        priority=4,
        name="Analysis Features",
        description="Provide summaries, comparisons, trend analysis, and highlight anomalies."
    ),
    Goal(
        priority=5,
        name="Reference Sources",
        description="Always reference the file(s) and section(s) used in the answer."
    ),
    Goal(
        priority=6,
        name="Clarifying Questions",
        description="Ask clarifying questions if the data or request is unclear."
    )
]



class Action:
    def __init__(self,
                 name: str,
                 function: Callable,
                 description: str,
                 parameters: Dict,
                 terminal: bool = False):
        self.name = name
        self.function = function
        self.description = description
        self.terminal = terminal
        self.parameters = parameters

    def execute(self, **args) -> Any:
        """Execute the action's function"""
        return self.function(**args)


class ActionRegistry:
    def __init__(self):
        self.actions = {}

    def register(self, action: Action):
        self.actions[action.name] = action

    def get_action(self, name: str) -> [Action, None]:
        return self.actions.get(name, None)

    def get_actions(self) -> List[Action]:
        """Get all registered actions"""
        return list(self.actions.values())
    

def list_files() -> list:
    """
    Lists all CSV and Excel files in the specified directory.
    """
    path = "/Users/mahdihanifi/Documents/GitHub/AI-Agents-and-Agentic-AI-with-Pytho-and-Generative-AI/documents"
    path = os.path.join(path)
    try:
        files = os.listdir(path)
        # Filter for CSV and Excel files
        data_files = [f for f in files if f.endswith('.csv') or f.endswith('.xlsx')]
        return data_files
    except FileNotFoundError:
        return f"Error: Directory not found at {path}"
    except Exception as e:
        return f"An error occurred: {e}"
    



    
def read_data(file_name: str) -> pd.DataFrame:
    """
    Reads a CSV or Excel file from a given file path into a pandas DataFrame.
    Handles errors for non-existent files or unsupported formats.
    """
    path =  "/Users/mahdihanifi/Documents/GitHub/AI-Agents-and-Agentic-AI-with-Pytho-and-Generative-AI/documents"
    file = os.path.join(path, file_name)
    try:
        # Check the file extension to use the correct pandas function
        if file.endswith('.csv'):
            df = pd.read_csv(file)
            return df.to_dict(orient="records")
        elif file.endswith('.xlsx'):
            df = pd.read_excel(file)
            return df.to_dict(orient="records")
        else:
            # If the format isn't supported, we return an error message
            return "Error: Unsupported file format. Please use .csv or .xlsx."
    except FileNotFoundError:
        # If the file doesn't exist, we return an error message
        return f"Error: File not found at {file}"
    except Exception as e:
        # Catch any other potential errors during file reading
        return f"An error occurred: {e}"
    

def analyze_data(df: pd.DataFrame, command: str):
    """
    Dynamically executes a pandas command on a DataFrame.
    Example command: "df['Sale_Amount'].mean()"
    """
    try:
        # The eval() function runs the code in the command string
        result = eval(command)
        return result
    except Exception as e:
        # If the command is invalid, return an error message
        return f"Error executing command: {e}"
# Create and populate the action registry
registry = ActionRegistry()

registry.register(Action(
    name="list_files",
    function=list_files,
    description="Lists all CSV and Excel files in the specified directory.",
    parameters={
        "type": "object",
        "properties": {},
        "required": []
    },
    terminal=False
))



registry.register(Action(
    name="read_data",
    function=read_data,
    description="Read a CSV or Excel file into a pandas DataFrame.",
    parameters={
        "type": "object",
        "properties": {
            "file_name": {
                "type": "string",
                "description": "Name of the file to read"
            }
        },
        "required": ["file_name"]
    },
    terminal=False
))  

registry.register(Action(
    name="analyze_data",
    function=analyze_data,
    description="Analyze the data using a pandas command.",
    parameters={
        "type": "object",
        "properties": {
            "df": {
                "type": "array",
                "description": "DataFrame data as a list of records",
                "items": {
                    "type": "object",
                    "additionalProperties": True
                }
            },
            "command": {
                "type": "string",
                "description": "Pandas command to execute on the DataFrame"
            }
        },
        "required": ["df", "command"]
    },
    terminal=False
))


########################This action will terminate the agent loop########################
registry.register(Action(
    name="terminate",
    function=lambda message=None: {"message": message or "Agent terminated by user."},
    description="Terminate the agent loop when requested by user.",
    parameters={
        "type": "object",
        "properties": {
            "message": {"type": "string"}
        },
        "required": []
    },
    terminal=True
))
###########################################################################################

class Memory:
    def __init__(self):
        self.items = []  # Basic conversation histor

    def add_memory(self, memory: dict):
        """Add memory to working memory"""
        self.items.append(memory)

    def get_memories(self, limit: int = None) -> List[Dict]:
        """Get formatted conversation history for prompt"""
        return self.items[:limit]

    def copy_without_system_memories(self):
        """Return a copy of the memory without system memories"""
        filtered_items = [m for m in self.items if m["type"] != "system"]
        memory = Memory()
        memory.items = filtered_items
        return memory


class Environment:
    def execute_action(self, action: Action, args: dict) -> dict:
        """Execute an action and return the result."""
        try:
            result = action.execute(**args)
            return self.format_result(result)
        except Exception as e:
            return {
                "tool_executed": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def format_result(self, result: Any) -> dict:
        """Format the result with metadata."""
        return {
            "tool_executed": True,
            "result": result,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")
        }



class AgentLanguage:
    def __init__(self):
        pass

    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory) -> Prompt:
        raise NotImplementedError("Subclasses must implement this method")


    def parse_response(self, response: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method")



class AgentFunctionCallingActionLanguage(AgentLanguage):

    def __init__(self):
        super().__init__()

    def format_goals(self, goals: List[Goal]) -> List:
        # Map all goals to a single string that concatenates their description
        # and combine into a single message of type system
        sep = "\n-------------------\n"
        goal_instructions = "\n\n".join([f"{goal.name}:{sep}{goal.description}{sep}" for goal in goals])
        return [
            {"role": "system", "content": goal_instructions}
        ]

    def format_memory(self, memory: Memory) -> List:
        """Generate response from language model"""
        # Map all environment results to a role:user messages
        # Map all assistant messages to a role:assistant messages
        # Map all user messages to a role:user messages
        items = memory.get_memories()
        mapped_items = []
        for item in items:

            content = item.get("content", None)
            if not content:
                content = json.dumps(item, indent=4)

            if item["type"] == "assistant":
                mapped_items.append({"role": "assistant", "content": content})
            elif item["type"] == "environment":
                mapped_items.append({"role": "assistant", "content": content})
            else:
                mapped_items.append({"role": "user", "content": content})

        return mapped_items

    def format_actions(self, actions: List[Action]) -> [List,List]:
        """Generate response from language model"""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": action.name,
                    # Include up to 1024 characters of the description
                    "description": action.description[:1024],
                    "parameters": action.parameters,
                },
            } for action in actions
        ]

        return tools

    def construct_prompt(self,
                         actions: List[Action],
                         environment: Environment,
                         goals: List[Goal],
                         memory: Memory) -> Prompt:

        prompt = []
        prompt += self.format_goals(goals)
        prompt += self.format_memory(memory)

        tools = self.format_actions(actions)

        return Prompt(messages=prompt, tools=tools)

    def adapt_prompt_after_parsing_error(self,
                                         prompt: Prompt,
                                         response: str,
                                         traceback: str,
                                         error: Any,
                                         retries_left: int) -> Prompt:

        return prompt

    def parse_response(self, response: str) -> dict:
        """Parse LLM response into structured format by extracting the ```json block"""

        try:
            return json.loads(response)

        except Exception as e:
            return {
                "tool": "terminate",
                "args": {"message":response}
            }



class Agent:
    def __init__(self,
                 goals: List[Goal],
                 agent_language: AgentLanguage,
                 action_registry: ActionRegistry,
                 generate_response: Callable[[Prompt], str],
                 environment: Environment):
        """
        Initialize an agent with its core GAME components
        """
        self.goals = goals
        self.generate_response = generate_response
        self.agent_language = agent_language
        self.actions = action_registry
        self.environment = environment

    def construct_prompt(self, goals: List[Goal], memory: Memory, actions: ActionRegistry) -> Prompt:
        """Build prompt with memory context"""
        return self.agent_language.construct_prompt(
            actions=actions.get_actions(),
            environment=self.environment,
            goals=goals,
            memory=memory
        )

    def get_action(self, response):
        invocation = self.agent_language.parse_response(response)
        action = self.actions.get_action(invocation["tool"])
        return action, invocation

    def should_terminate(self, response: str) -> bool:
        action_def, _ = self.get_action(response)
        return action_def.terminal

    def set_current_task(self, memory: Memory, task: str):
        memory.add_memory({"type": "user", "content": task})

    def update_memory(self, memory: Memory, response: str, result: dict):
        """
        Update memory with the agent's decision and the environment's response.
        """
        new_memories = [
            {"type": "assistant", "content": response},
            {"type": "environment", "content": json.dumps(result)}
        ]
        for m in new_memories:
            memory.add_memory(m)

    def prompt_llm_for_action(self, full_prompt: Prompt) -> str:
        response = self.generate_response(full_prompt)
        return response

    def run(self, user_input: str, memory=None, max_iterations: int = 50) -> Memory:
        """
        Execute the GAME loop for this agent with a maximum iteration limit.
        """
        memory = memory or Memory()
        self.set_current_task(memory, user_input)

        for _ in range(max_iterations):
            # Construct a prompt that includes the Goals, Actions, and the current Memory
            prompt = self.construct_prompt(self.goals, memory, self.actions)

            print("Agent thinking...")
            # Generate a response from the agent
            response = self.prompt_llm_for_action(prompt)
            print(f"Agent Decision: {response}")

            # Determine which action the agent wants to execute
            action, invocation = self.get_action(response)


            # Safety check for unregistered actions
            if action is None:
                print(f"No action found for tool: {invocation.get('tool')}")
                # Treat as a non-terminal user message, cintinue the loop
                memory.add_memory({"type": "user", "content": response})
                continue    

            # Execute the action in the environment
            result = self.environment.execute_action(action, invocation["args"])
            print(f"Action Result: {result}")

            # Update the agent's memory with information about what happened
            self.update_memory(memory, response, result)

            # Check if the agent has decided to terminate
            if self.should_terminate(response):
                print("Agent terminated.")
                break

        return memory
    


environment = Environment()
agent_language = AgentFunctionCallingActionLanguage()
action_registry = registry
environment = Environment()


    
# Create the agent
file_explorer_agent = Agent(
    goals=goals,
    agent_language=agent_language,
    action_registry=action_registry,
    generate_response=generate_response,
    environment=environment
)

# Run the agent
user_input = input("What would you like me to do? ")
final_memory = file_explorer_agent.run(user_input, max_iterations=10)

# Print the final conversation if desired
for item in final_memory.get_memories():
    print(f"\n{item['type'].upper()}: {item['content']}")
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable


Model="qwen3:1.7b"
MAX_ITERATIONS=10

@tool 
def get_cricketer_runs(name:str)->int:
    """Get total number of runs scored by cricketer"""
    print(f"Getting the runs scored by cricketer {name}...")
    runs={"Sachin":30000, "Virat":24000, "Dhoni":21000, "Rohit":12000}
    return runs.get(name,0)


@tool
def get_number_of_matches_played(name:str)->int:
    """Get the number of matches played by the cricketr"""
    print(f"Getting the number of matches played by {name}...")
    matches={"Sachin":200, "Virat":130, "Dhoni":120, "Rohit":150}
    return matches.get(name,0)

@tool
def calculate_avg_for_cricketer(total_runs:int, number_of_matches:int)->float:
    """Calculate the average runs scored by crickets per match"""
    if number_of_matches==0:
        return 0
    return round(total_runs/number_of_matches,2)

@traceable(name="Cricket Stats Trace")
def run_agent(question:str):
    tools=[get_cricketer_runs, get_number_of_matches_played, calculate_avg_for_cricketer]
    tools_dict={tool.name: tool for tool in tools}
    llm=init_chat_model(f"ollama:{Model}",temperature=0.7)
    #llm=init_chat_model(f"openai:gpt-5.2",temperature=0.7)
    llm_with_tools=llm.bind_tools(tools)
    print(f"Question: {question}")
    print("*"*80)

    messages = [
                SystemMessage(
                    content = "You are a helpful assistant that provides the average runs scored by a cricketer. \n"
                    "STRICT RULES - you must follow these exactly: \n"
                    "1. You will need values of total runs and number of matches played to get the average runs. \n"
                    "2. You have access to tool get_cricketer_runs to get the total runs scored by a cricketer. \n"
                    "3. You can calculate average runs scored by cricketer if you have matches and total runs. \n"
                    "4. You have access to tool calculate_avg_for_cricketer to calculate the average runs scored by a cricketer per match. You will need total runs and number of matches to calculate. \n"
                    "5. You have access to tool get_number_of_matches_played to get the number of matches played by the cricketer. \n"
                    "6. Always use the tools provided to answer the question. \n"
                    "7. Never assume and guess the runs scored by the cricketer - you MUST use the get_cricketer_runs tool to get the runs. \n"
                    "8. Never assume the number of matches played by the cricketer - you MUST ask the user for the number of matches played."
                    ),
                 HumanMessage(content = question)
    ]


    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"Iteration:{iteration}")
        ai_message = llm_with_tools.invoke(messages)
        tools_used = ai_message.tool_calls

        if not tools_used:
            print(f"\nAI response:{ai_message.content}")
            return ai_message.content

        tool_call = tools_used[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")
        print(f"Tool called: {tool_name} with args: {tool_args}")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None: 
            raise ValueError(f"Tool {tool_name} not found in tools_dict")

        observation = tool_to_use.invoke(tool_args)
        print(f"Observation from tool: {observation}")

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("Max iterations reached without a final answer.")
    return None
    
if __name__=="__main__":
    question="What is the average runs scored by Virat?"
    run_agent(question)

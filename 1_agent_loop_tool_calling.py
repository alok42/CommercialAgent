from dotenv import load_dotenv


load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10

Model= "qwen3:1.7b"

#-----Tools (Langchain @tool decorator)-----

@tool
def get_product_price(product: str) ->float:
    """Get the prices of the product"""
    print(f"Getting price for {product}...")
    prices={"laptop": 1000, "headphones":200, "phone": 230, "keyboard":150}
    return prices.get(product,0)

@tool
def apply_discount(price: float, discount_tier:str)->float:
    """Apply discount to the price based on the discount tier"""
    discount_rates={"gold":23, "bronze":5, "silver":10}
    discount = discount_rates.get(discount_tier,0)
    return round(price * (1 - discount / 100), 2)

#-----Agent Loop----- 

@traceable(name="Commercial Agent traces")
def run_agent(question:str):
    tools=[get_product_price, apply_discount]
    tools_dict={tool.name: tool for tool in tools}

    #llm=init_chat_model(f"openai:gpt-5.2",temperature=0.7)
    llm = init_chat_model(f"ollama:{Model}",temperature=0.7)
    llm_with_tools=llm.bind_tools(tools)

    print(f"Question: {question}")
    print("="*60)

    messages = [
        SystemMessage(
            content="You are a helpful assistant."
            "You have access to a Product catalog and a dicount tool.\n\n"
            "STRICT RULES - you must follow these exactly: \n"
            "1. Never guess or assume any product price"
            "You MUST call get_product_price tool to get the price of the product. \n"
            "2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price \n"
            "return by get_product_price - do NOT pass a made-up number. \n"
            "3. Never calculate discounts yourself using math. "
            "Always use the apply_discount tool.\n"
            "4. If the user does not specify the discount tier, ask them which tier to use - do NOT assume one"
            "5. DO NOT assume currency unless specified by the user. \n" 
            "6. Always use the tools when needed - do NOT provide final answers without using the tools. \n"
            "7. Aways get the price and product info from tools"
            ),
        HumanMessage(content=question),
    ]

    for iteration in range (1, MAX_ITERATIONS+1):
        print(f"\n----Iteration {iteration}----")
        ai_message=llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nAI response:{ai_message.content}")
            return ai_message.content
        
        #Process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args",{})
        tool_call_id = tool_call.get("id")

        print(f"\n [Tool selected] {tool_name} with args {tool_args}")   
        tool_to_use = tools_dict.get(tool_name)

        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in available tools.")
        
        observation = tool_to_use.invoke(tool_args)
        print(f"\n [Tool observation] {observation}")
        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Max iterations reached without a final answer.")
    return None






if __name__ == "__main__":
    print("Welcome to the Commercial Agent!")
    print()
    result=run_agent("What is the price of a laptop with a gold tier discount?")
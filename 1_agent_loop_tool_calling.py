from dotenv import load_dotenv

from asyncio import tools
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

llm = init_chat_model(f"ollama:{Model}",temperature=0.7)
llm_with_tools=llm.bind_tools(tools)

print(f"Question: {question}")
print("="*60)

if __name__ == "__main__":
    print("Welcome to the Commercial Agent!")
    print()
    result=run_agent("What is the price of a laptop with a gold discount?")


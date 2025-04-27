import asyncio

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from mcp_use import MCPAgent, MCPClient
import os

async def run_memory_chat():
    
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    config_file = "browser_mcp.json"

    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model = "qwen-qwq-32b")

    agent = MCPAgent(
        llm = llm,
        client = client,
        max_steps = 15,
        memory_enabled = True,
        verbose = True
    )

    print("=======CHAT START=======")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("=======CHAT END=======")
                break
            if user_input.lower() in ["clear", "reset"]:
                agent.clear_conversation_history()
                print("=======MEMORY CLEARED=======")
                continue
            
            print("\nAssistant: ", end="", flush=True)
            try:
                response = await agent.run(user_input)
                print(f"{response}\n")
            except Exception as e:
                print(f"Error: {e}")
    finally:
        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    print("=======CHAT START=======")
    asyncio.run(run_memory_chat())
        
            
    


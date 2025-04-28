
import os
from dotenv import load_dotenv

load_dotenv()

groq_api = os.getenv("GROQ_API_KEY")
tavily_api = os.getenv("TAVILY_API_KEY")

if not groq_api:
    print("Warning: GROQ_API_KEY not found in .env file or environment variables.")

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage


groq_llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api)

def get_response(query, system_prompt, allow_search):

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    # system_prompt = "You are a tour guide and planner, who loves to keep record of the numbers in tour planning."

    agent = create_react_agent(
        model=groq_llm,
        tools=tools,
        state_modifier=system_prompt
    )

    # query = "Plan my tour from Karachi to Tharparkar by road, include roads to travel from, some good spots to stop by while traveling and cost  in the plan."

    state = {"messages": query}

    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]

    return ai_messages[-1]
 

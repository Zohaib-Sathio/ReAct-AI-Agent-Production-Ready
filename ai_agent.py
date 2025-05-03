
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
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.memory import ConversationEntityMemory



# Short-term memroy saver
checkpointer = InMemorySaver()

# Long-term memory store
store = InMemoryStore(index={"embed": lambda x: x, "dims":1536})

# LLM
groq_llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api)

# Entity memory
entity_memory = ConversationEntityMemory(llm=groq_llm)

def extract_and_store_entities(state):
    print("Entity memory being updated.")
    """
    Tool to update entity memory from latest turn.
    """
    last = state["messages"][-1]
    # Save context into entity memory
    entity_memory.save_context({"input": last}, {"output": last})
    return {}


tools = [TavilySearchResults(max_results=2), extract_and_store_entities]


agent = create_react_agent(
        model=groq_llm,
        tools=tools,
        checkpointer=checkpointer,
        store=store
    )


def get_response(query, system_prompt, thread_id):
    print("Checkpointer: ", checkpointer)
    print("Store: ", store)
    print("Entity Memory: ", entity_memory)

    # system_prompt = "You are a tour guide and planner, who loves to keep record of the numbers in tour planning."


    # query = "Plan my tour from Karachi to Tharparkar by road, include roads to travel from, some good spots to stop by while traveling and cost  in the plan."

    human_msgs = [HumanMessage(content=m) for m in query]

    result = agent.invoke(
        {"messages": human_msgs},
        config={"configurable":{"thread_id": thread_id, "system_prompt": system_prompt}}
    )


    messages = result.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]

    return ai_messages[-1]
 

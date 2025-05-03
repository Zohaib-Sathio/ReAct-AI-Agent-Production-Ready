from pydantic import BaseModel
from typing import List
from ai_agent import get_response

class RequestState(BaseModel):
    thread_id: str
    system_prompt: str
    messages: List[str]


from fastapi import FastAPI

app = FastAPI(title="LangGraph AI Agent")

@app.post("/get_response")
def response_endpoint(request_state: RequestState):
    """
    FastAPI endpoint which is used to get response from the agent with short‑term, long‑term, and entity memory.
    """
    response = get_response(request_state.messages, request_state.system_prompt, request_state.thread_id)
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9090)

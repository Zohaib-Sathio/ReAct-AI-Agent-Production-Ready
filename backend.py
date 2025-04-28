from pydantic import BaseModel
from typing import List
from ai_agent import get_response

class RequestState(BaseModel):
    system_prompt: str
    messages: List[str]
    allow_search: bool


from fastapi import FastAPI

app = FastAPI(title="LangGraph AI Agent")

@app.post("/get_response")
def response_endpoint(request_state: RequestState):
    """
    FastAPI endpoint which is used to get response from the agent. 
    """
    response = get_response(request_state.messages, request_state.system_prompt, request_state.allow_search)
    return {"response", response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9090)

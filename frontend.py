import streamlit as st
import requests
import json

st.set_page_config(page_title="LangGraph ReAct Agent UI", layout="centered")
st.title("ReAct AI Agent")
st.write("Write agent role and query and Interact with it.")

# Added field for thread-scoped memory identifier
thread_id = st.text_input("Thread ID (for memory):", value="user-zohaib")


system_prompt = st.text_area("Define your Agent role: ", height=75, value="You are a tour guide and planner, who loves to keep record of the numbers in tour planning.")

user_query = st.text_area("Enter your query: ", height=140, value="Plan my tour from Karachi to Tharparkar by road on a motorbike, include roads to travel from, estimated time to reach destination, some good spots to stop by while traveling and cost in the plan. Present these details in a tabular form.")

BACKEND_URL = "http://127.0.0.1:9090/get_response"
if st.button("Run Agent!"):
    if user_query.strip():
        payload = {
            "messages": [user_query],
            "system_prompt": system_prompt,
            "thread_id": thread_id

        }
        try:
            with st.spinner("Agent is thinking..."):
                api_response = requests.post(
                    BACKEND_URL,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload)
                )

                api_response.raise_for_status()
                if api_response.status_code == 200:
                    response_data = api_response.json()

                    st.subheader("Agent Response:")
                    st.markdown(response_data)
                else:
                    st.error(f"Request failed with status code: {api_response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a query.")

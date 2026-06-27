import streamlit as st
import requests

# Point this to your locally running FastAPI server
API_URL = "http://127.0.0.1:8000/ask"

# Configure the page
st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🏢", layout="centered")
st.title("🏢 Enterprise Knowledge Assistant")
st.markdown("Ask questions about company policies, technical guides, and internal processes.")

# --- Bonus Feature: Conversation Memory ---
# We use Streamlit's session_state to remember the chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-draw all past messages to the screen when the user interacts
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the message is from the assistant, also draw the source expander
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"🔍 View Sources (Confidence: {message['confidence']})"):
                for idx, source in enumerate(message["sources"]):
                    st.write(f"{idx + 1}. **{source['document']}** (Page {source['page']})")

# --- Chat Input & API Interaction ---
if prompt := st.chat_input("e.g., What is the employee leave policy?"):
    
    # 1. Display and save the user's question
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Display the assistant's response container
    with st.chat_message("assistant"):
        with st.spinner("Searching the enterprise vault..."):
            try:
                # Call your FastAPI backend
                response = requests.post(API_URL, json={"question": prompt})
                response.raise_for_status() # Catch HTTP errors (e.g., 500 or 404)
                
                # Parse the JSON response
                data = response.json()
                answer = data.get("answer", "Information unavailable.")
                sources = data.get("sources", [])
                confidence = data.get("confidence", 0.0)

                # Render the answer
                st.markdown(answer)
                
                # Render the collapsible source references
                if sources:
                    with st.expander(f"🔍 View Sources (Confidence: {confidence})"):
                        for idx, source in enumerate(sources):
                            st.write(f"{idx + 1}. **{source['document']}** (Page {source['page']})")
                
                # Save the assistant's response to memory
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence
                })

            except requests.exceptions.RequestException as e:
                st.error(f"⚠️ API Connection Error: Ensure your FastAPI server is running. Details: {e}")
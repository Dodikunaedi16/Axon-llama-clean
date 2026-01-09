import os
import streamlit as st
import replicate

# ======================
# Page configuration
# ======================
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot by Esmail Gumaan")

# ======================
# Load Replicate API Token (ENV ONLY)
# ======================
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    st.error("REPLICATE_API_TOKEN belum diset. Gunakan file .env atau environment variable.")
    st.stop()

replicate.api_token = REPLICATE_API_TOKEN

# ======================
# Sidebar
# ======================
with st.sidebar:
    st.title("🦙💬 Llama 2 Chatbot")
    st.title("👨‍💻🤖: Esmail A. Gumaan")
    st.write(
        "This chatbot is created using the open-source Llama 2 LLM model from Meta."
    )

    st.subheader("Models and parameters")

    selected_model = st.selectbox(
        "Choose a Llama2 model",
        ["Llama2-7B", "Llama2-13B"],
    )

    if selected_model == "Llama2-7B":
        llm = "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea"
    else:
        llm = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"

    temperature = st.slider("temperature", 0.01, 1.0, 0.1, 0.01)
    top_p = st.slider("top_p", 0.01, 1.0, 0.9, 0.01)
    max_length = st.slider("max_length", 32, 128, 120, 8)

# ======================
# Chat state
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

# ======================
# Llama2 Response Generator
# ======================
def generate_llama2_response(prompt_input):
    dialogue = (
        "You are a helpful assistant. "
        "Do not respond as 'User'. Respond only once as 'Assistant'.\n\n"
    )

    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        dialogue += f"{role}: {msg['content']}\n\n"

    response = replicate.run(
        llm,
        input={
            "prompt": f"{dialogue}User: {prompt_input}\n\nAssistant:",
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
            "repetition_penalty": 1,
        },
    )
    return response


# ======================
# User input
# ======================
if prompt := st.chat_input("Type your message here…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# ======================
# Generate response
# ======================
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ""

            for chunk in response:
                full_response += chunk
                placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

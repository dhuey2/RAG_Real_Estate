import openai
import streamlit as st

# Set API Key
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found! Please add it to Streamlit secrets.")
else:
    openai.api_key = OPENAI_API_KEY
client = openai.OpenAI(api_key=OPENAI_API_KEY)



def generate_answer_with_openai(query, context_chunks):
    """
    Generates an answer using OpenAI's ChatGPT model with the new v1.0 API format.
    """
    context = "\n\n".join(context_chunks)

    messages = [
        {"role": "system", "content": "You are an intelligent assistant with expertise in real estate. Your goal is to help answer queries about real estate documents and explain them in simpler terms. Use the provided context to answer the user's query."},
        {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if needed
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while generating the response."


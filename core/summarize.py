import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()



def get_llm():
    import streamlit as st

    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Check your local .env file "
            "or Streamlit Cloud Secrets."
        )

    return ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0.2,
    max_retries=3,
    timeout=60,
)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    if not transcript or not transcript.strip():
        raise ValueError("Transcript cannot be empty.")

    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Summarize this portion of a meeting transcript concisely. "
            "Preserve important points, facts, and decisions. "
            "Do not invent information."
        ),
        ("human", "{text}")
    ])

    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)

    chunk_summaries = [
        map_chain.invoke({"text": chunk})
        for chunk in chunks
    ]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Combine these partial summaries into one professional "
            "meeting summary in clear bullet points. Remove repetition "
            "and preserve important decisions and action items."
        ),
        ("human", "{text}")
    ])

    combined_chain = combined_prompt | llm | StrOutputParser()

    return combined_chain.invoke({"text": combined})


def generate_title(transcript: str) -> str:
    if not transcript or not transcript.strip():
        return "Untitled Meeting"

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate a professional meeting title in no more than "
            "8 words. Return only the title."
        ),
        ("human", "{text}")
    ])

    title_chain = title_prompt | llm | StrOutputParser()

    return title_chain.invoke(
        {"text": transcript[:2000]}
    ).strip()

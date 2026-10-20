import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def get_llm():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.2
    )


def build_chain(system_prompt: str):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    return prompt | llm | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. Extract all action items "
        "explicitly stated or clearly implied in the transcript.\n"
        "For each action item provide:\n"
        "- Task description\n"
        "- Owner (use 'Not specified' if unknown)\n"
        "- Deadline (use 'Not specified' if unknown)\n"
        "Do not invent tasks, owners, or deadlines. "
        "Format as a numbered list. "
        "If none are found, say 'No action items found.'"
    )

    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. Extract the key decisions "
        "actually made during the meeting. Distinguish confirmed decisions "
        "from suggestions or proposals. Do not invent decisions. "
        "Format as a numbered list. "
        "If none are found, say 'No key decisions found.'"
    )

    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. Extract unresolved questions, "
        "uncertainties, and topics requiring follow-up from the transcript. "
        "Do not include questions that were clearly answered. "
        "Do not invent questions. Format as a numbered list. "
        "If none are found, say 'No open questions found.'"
    )

    return chain.invoke({"text": transcript})


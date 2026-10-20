
import streamlit as st
from pathlib import Path

from main import run_pipeline
from core.rag_engine import ask_question


st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ AI Video & Meeting Assistant")
st.caption(
    "Turn long videos and meetings into summaries, "
    "action items, decisions, and searchable knowledge."
)

# Store results between Streamlit reruns
if "meeting_result" not in st.session_state:
    st.session_state.meeting_result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("⚙️ Meeting Settings")

    language = st.selectbox(
        "Transcription language",
        ["english", "hinglish"],
        format_func=lambda x: x.title(),
    )

    st.divider()

    st.info(
        "Upload an audio/video file or paste a YouTube URL. "
        "Long videos may take several minutes to process."
    )


# ---------------- INPUT SECTION ----------------

st.subheader("📥 Add a Meeting")

youtube_url = st.text_input(
    "YouTube URL (optional)",
    placeholder="https://www.youtube.com/watch?v=...",
)

uploaded_file = st.file_uploader(
    "Or upload an audio/video file",
    type=[
        "mp3", "wav", "m4a", "mp4",
        "mpeg", "webm", "ogg", "mpga"
    ],
)

if st.button("🚀 Process Meeting", type="primary"):
    if not youtube_url.strip() and uploaded_file is None:
        st.warning("Please enter a YouTube URL or upload a file.")

    else:
        try:
            if uploaded_file is not None:
                upload_dir = Path("data/uploads")
                upload_dir.mkdir(parents=True, exist_ok=True)

                file_path = upload_dir / Path(
                    uploaded_file.name
                ).name

                file_path.write_bytes(uploaded_file.getvalue())
                source = str(file_path.resolve())
            else:
                source = youtube_url.strip()

            with st.spinner(
                "Processing meeting... This may take a while."
            ):
                result = run_pipeline(source, language)

            st.session_state.meeting_result = result
            st.session_state.chat_history = []

            st.success("Meeting processed successfully!")
            st.rerun()

        except Exception as exc:
            st.error(
                "Meeting processing failed. Check the terminal "
                "for the full error details."
            )
            st.code(str(exc))
            st.info(
                "If the error says 429 RESOURCE_EXHAUSTED, "
                "your Gemini API quota is exhausted. "
                "Wait for quota reset or configure another model."
            )


# ---------------- DISPLAY RESULTS ----------------

result = st.session_state.meeting_result

if result:
    st.divider()

    st.header(f"📌 {result['title']}")

    summary_tab, actions_tab, decisions_tab, questions_tab, transcript_tab = (
        st.tabs([
            "📝 Summary",
            "✅ Action Items",
            "🔑 Decisions",
            "❓ Open Questions",
            "📄 Transcript",
        ])
    )

    with summary_tab:
        st.markdown(result["summary"])

    with actions_tab:
        st.markdown(result["action_items"])

    with decisions_tab:
        st.markdown(result["key_decisions"])

    with questions_tab:
        st.markdown(result["open_questions"])

    with transcript_tab:
        transcript = result["transcript"]
        st.text_area(
            "Full transcription",
            value=transcript,
            height=350,
        )

        st.download_button(
            "⬇️ Download Transcript",
            data=transcript,
            file_name="meeting_transcript.txt",
            mime="text/plain",
        )

    # ---------------- CHAT WITH MEETING ----------------

    st.divider()
    st.header("💬 Chat with Your Meeting")
    st.caption(
        "Ask questions about the transcript and its contents."
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask something about this meeting..."
    )

    if prompt:
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt,
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Finding the answer..."):
                    answer = ask_question(
                        result["rag_chain"],
                        prompt,
                    )
                st.markdown(answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": str(answer),
            })

        except Exception as exc:
            st.error(
                "Could not answer your question. "
                "Check the terminal for details."
            )
            st.code(str(exc))

else:
    st.info(
        "Your meeting results will appear here after processing."
    )

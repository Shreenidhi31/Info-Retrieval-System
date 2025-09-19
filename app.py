import streamlit as st
import time
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def send_user_question(user_question):
    """Send user question to the conversational chain and update chat history in session_state."""
    if st.session_state.conversation is None:
        st.error("Please upload PDFs and click Submit & Process first.")
        return

    # call chain with the expected input key "question"
    result = st.session_state.conversation({"question": user_question})
    # ConversationalRetrievalChain returns a dict like {"answer": "...", "source_documents": [...]}
    answer = result.get("answer") or result.get("result") or str(result)

    # append to chat history (list of (role, text))
    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("assistant", answer))


def main():
    st.set_page_config(page_title="Information Retrieval")
    st.header("INFORMATION RETRIEVAL SYSTEM")

    # initialize session state keys (consistent names)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list of tuples ("user"/"assistant", text)

    # left: sidebar for upload and processing
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and click Submit & Process",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    time.sleep(1)  # optional small delay
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("Processing done — you can ask questions now!")

    # main area: user input + display
    user_question = st.text_input("Ask a question from the uploaded PDFs", key="input_box")

    if st.button("Ask"):
        if user_question:
            send_user_question(user_question)
            # clear input box
            st.session_state.input_box = ""

    # display chat history
    st.subheader("Chat")
    if st.session_state.chat_history:
        for role, text in st.session_state.chat_history:
            if role == "user":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**Bot:** {text}")
    else:
        st.info("No conversation yet. Upload PDFs and click Submit & Process to start.")

if __name__ == "__main__":
    main()

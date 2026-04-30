import streamlit as st

st.title("🎥 YouTube RAG with Whisper Fallback")

url = st.text_input("Paste YouTube Link")

if url:

    with st.spinner("Processing video..."):

        # STEP 1: try transcript first (optional)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            video_id = url.split("v=")[-1]
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([t["text"] for t in transcript])

        except:
            st.warning("Transcript not found. Using Whisper...")

            audio_file = download_audio(url)
            text = audio_to_text(audio_file)

        # STEP 2: build RAG
        db = build_vector_db(text)
        retriever = db.as_retriever(search_kwargs={"k": 4})

        st.success("Ready! Ask questions below 👇")

        query = st.chat_input("Ask something")

        if query:
            answer = get_answer(query, retriever)

            st.chat_message("user").write(query)
            st.chat_message("assistant").write(answer)

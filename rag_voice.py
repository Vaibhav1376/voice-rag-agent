"""
Voice RAG Agent — Conversational Documentation Intelligence Platform
Built with OpenAI Agents SDK, Qdrant Vector Database, FastEmbed, and Streamlit.
"""

from typing import List, Dict, Optional, Tuple
import os
import tempfile
from datetime import datetime
import uuid
import asyncio

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from fastembed import TextEmbedding
from openai import AsyncOpenAI

# Load environment configuration
load_dotenv()

# ==============================================================================
# Agent Framework Configuration & Fallback Adapter
# ==============================================================================
try:
    from agents import Agent, Runner
    HAS_OPENAI_AGENTS = True
except Exception:
    HAS_OPENAI_AGENTS = False

    class Agent:
        """Lightweight adapter compatible with OpenAI Agents SDK interface."""
        def __init__(self, name: str, instructions: str, model: str = "gpt-4o"):
            self.name = name
            self.instructions = instructions
            self.model = model

    class Runner:
        """Lightweight execution engine routing agent instructions through OpenAI API."""
        @staticmethod
        async def run(agent: Agent, prompt: str):
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = await client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": agent.instructions},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            class AgentResult:
                def __init__(self, output: str):
                    self.final_output = output
            return AgentResult(response.choices[0].message.content or "")

# Local Audio Player import check
try:
    from openai.helpers import LocalAudioPlayer
    HAS_LOCAL_AUDIO = True
except ImportError:
    HAS_LOCAL_AUDIO = False

# Constants
COLLECTION_NAME = "voice-rag-agent"
DEFAULT_VOICES = ["coral", "alloy", "ash", "ballad", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]


def init_session_state() -> None:
    """Initialize Streamlit session state with environment defaults."""
    env_voice = os.getenv("DEFAULT_VOICE", "coral")
    default_voice = env_voice if env_voice in DEFAULT_VOICES else "coral"

    defaults = {
        "initialized": False,
        "qdrant_url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "tts_agent": None,
        "selected_voice": default_voice,
        "processed_documents": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def setup_sidebar() -> None:
    """Configure sidebar with API credentials, voice persona settings, and system status."""
    with st.sidebar:
        st.title("⚙️ System Configuration")
        st.markdown("Configure your vector database and OpenAI credentials below:")
        
        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url,
            help="Enter 'http://localhost:6333', your Qdrant Cloud URL, or ':memory:' for fast local in-memory storage."
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value=st.session_state.qdrant_api_key,
            type="password",
            help="Required for Qdrant Cloud clusters. Leave blank for local Docker without auth."
        )
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Required for GPT-4o Agent reasoning and Speech generation."
        )
        
        st.markdown("---")
        st.markdown("### 🎙️ Voice Persona")
        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            options=DEFAULT_VOICES,
            index=DEFAULT_VOICES.index(st.session_state.selected_voice) if st.session_state.selected_voice in DEFAULT_VOICES else 0,
            help="Select the vocal tone and timbre for spoken answers."
        )

        st.markdown("---")
        st.markdown("### 📊 System Status")
        db_status = "🟢 Connected" if st.session_state.client else "⚪ Not Connected"
        doc_count = len(st.session_state.processed_documents)
        st.markdown(f"**Vector Store**: {db_status}")
        st.markdown(f"**Indexed Documents**: `{doc_count}`")


def setup_qdrant() -> Tuple[QdrantClient, TextEmbedding]:
    """Initialize Qdrant vector database client and dense embedding model."""
    url = (st.session_state.qdrant_url or "").strip()
    api_key = (st.session_state.qdrant_api_key or "").strip()
    
    if url.lower() in [":memory:", "memory"]:
        client = QdrantClient(location=":memory:")
    elif api_key:
        client = QdrantClient(url=url, api_key=api_key)
    elif url:
        client = QdrantClient(url=url)
    else:
        # Fallback to local in-memory storage if URL not specified
        client = QdrantClient(location=":memory:")
    
    # Initialize FastEmbed dense embedding model
    embedding_model = TextEmbedding()
    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)
    
    try:
        # Check if collection already exists before attempting creation
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise e
    
    return client, embedding_model


def process_document(file) -> List[Document]:
    """Extract text from uploaded document (PDF or Text/Markdown) and chunk with metadata."""
    try:
        file_name = file.name.lower()
        documents = []

        if file_name.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            try:
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        elif file_name.endswith(('.txt', '.md', '.markdown')):
            content = file.getvalue().decode("utf-8", errors="ignore")
            documents = [Document(page_content=content, metadata={"source": file.name})]
        else:
            st.error("Unsupported file format. Please upload a PDF, Markdown, or TXT file.")
            return []

        # Add rich source metadata
        for doc in documents:
            doc.metadata.update({
                "source_type": "pdf" if file_name.endswith('.pdf') else "text",
                "file_name": file.name,
                "timestamp": datetime.now().isoformat()
            })
        
        # Recursive chunking for optimal semantic search
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 Document processing error: {str(e)}")
        return []


def store_embeddings(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    documents: List[Document],
    collection_name: str
) -> None:
    """Embed chunks and upsert vectors into Qdrant vector store."""
    for doc in documents:
        embedding = list(embedding_model.embed([doc.page_content]))[0]
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": doc.page_content,
                        **doc.metadata
                    }
                )
            ]
        )


def setup_agents(openai_api_key: str) -> Tuple[Agent, Agent]:
    """Initialize dual agents: Documentation Processor and Audio Script Specialist."""
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    processor_agent = Agent(
        name="Documentation Processor",
        instructions="""You are an expert technical documentation assistant. Your task is to:
        1. Analyze the provided retrieved documentation content carefully.
        2. Answer the user's question clearly, accurately, and concisely.
        3. Include concrete code examples or key details when available in the context.
        4. Explicitly cite the source files when referencing specific content.
        5. Maintain a natural, authoritative, yet conversational tone.
        6. Structure your response so it flows smoothly when spoken out loud.""",
        model="gpt-4o"
    )

    tts_agent = Agent(
        name="Text-to-Speech Script Specialist",
        instructions="""You are an expert voice synthesis coach. Your task is to:
        1. Optimize the technical answer for natural human speech delivery.
        2. Replace complex symbols or acronyms with phonetic phrasing where needed.
        3. Maintain proper conversational pacing and natural pauses.
        4. Keep the delivery engaging, professional, and friendly.
        5. Ensure clarity for listeners without losing technical precision.""",
        model="gpt-4o"
    )
    
    return processor_agent, tts_agent


async def generate_audio_stream(
    async_openai: AsyncOpenAI,
    voice: str,
    text: str,
    instructions: str
):
    """Generate audio stream with model and parameter fallback."""
    try:
        return await async_openai.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="pcm",
        )
    except Exception:
        # Fallback to standard OpenAI TTS endpoint
        return await async_openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="pcm",
        )


async def generate_audio_file(
    async_openai: AsyncOpenAI,
    voice: str,
    text: str,
    instructions: str
):
    """Generate downloadable MP3 audio with parameter fallback."""
    try:
        return await async_openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="mp3"
        )
    except Exception:
        return await async_openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3"
        )


async def process_query(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
    collection_name: str,
    openai_api_key: str,
    voice: str
) -> Dict:
    """Orchestrate query semantic retrieval, dual-agent reasoning, and audio generation."""
    try:
        st.info("🔍 Step 1: Generating query dense embedding & searching Qdrant vector store...")
        query_embedding = list(embedding_model.embed([query]))[0]
        
        search_response = client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=4,
            with_payload=True
        )
        
        search_results = search_response.points if hasattr(search_response, 'points') else []
        st.write(f"Retrieved `{len(search_results)}` matching chunks from vector database.")
        
        if not search_results:
            raise Exception("No relevant documentation chunks found matching your question.")
        
        st.info("📑 Step 2: Assembling semantic context and source references...")
        context = "Based on the following retrieved documentation:\n\n"
        for i, result in enumerate(search_results, 1):
            payload = result.payload or {}
            content = payload.get('content', '')
            source = payload.get('file_name', 'Unknown Source')
            context += f"--- Document {i} [{source}] ---\n{content}\n\n"
        
        context += f"\nUser Question: {query}\n\n"
        context += "Provide a clear, accurate, and concise answer suitable for both reading and spoken delivery."
        
        st.info("🤖 Step 3: Initializing Multi-Agent Orchestrator...")
        if not st.session_state.processor_agent or not st.session_state.tts_agent:
            processor_agent, tts_agent = setup_agents(openai_api_key)
            st.session_state.processor_agent = processor_agent
            st.session_state.tts_agent = tts_agent
        
        st.info("✍️ Step 4: Documentation Processor Agent generating technical answer...")
        processor_result = await Runner.run(st.session_state.processor_agent, context)
        text_response = processor_result.final_output
        
        st.info("🎙️ Step 5: Audio Script Agent optimizing delivery cadence...")
        tts_result = await Runner.run(st.session_state.tts_agent, text_response)
        voice_instructions = tts_result.final_output
        
        st.info("🔊 Step 6: Synthesizing real-time voice audio stream...")
        async_openai = AsyncOpenAI(api_key=openai_api_key)
        
        # Audio playback
        if HAS_LOCAL_AUDIO:
            try:
                stream_response = await generate_audio_stream(
                    async_openai, voice, text_response, voice_instructions
                )
                async with stream_response as stream:
                    await LocalAudioPlayer().play(stream)
            except Exception as player_err:
                st.caption(f"ℹ️ Local speaker playback bypassed ({player_err}). Audio ready in player below.")
        
        # Generate MP3 asset for browser playback and download
        audio_response = await generate_audio_file(
            async_openai, voice, text_response, voice_instructions
        )
        
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"voice_response_{uuid.uuid4().hex[:8]}.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_response.content)
        
        unique_sources = list({r.payload.get('file_name', 'Unknown Source') for r in search_results if r.payload})

        return {
            "status": "success",
            "text_response": text_response,
            "voice_instructions": voice_instructions,
            "audio_path": audio_path,
            "sources": unique_sources
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }


def main() -> None:
    """Streamlit web application entrypoint."""
    st.set_page_config(
        page_title="Voice RAG Agent — OpenAI SDK & Qdrant",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    setup_sidebar()
    
    st.title("🎙️ Voice RAG Agent")
    st.markdown(
        "**Conversational Documentation Intelligence** — Ask questions about your technical documentation "
        "and receive real-time, context-grounded voice explanations powered by **OpenAI Agents SDK**, "
        "**Qdrant Vector Database**, and **FastEmbed**."
    )
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📁 1. Document Ingestion")
        uploaded_file = st.file_uploader(
            "Upload technical documentation (PDF, TXT, or Markdown)",
            type=["pdf", "txt", "md"],
            help="Upload your document to generate semantic vector embeddings."
        )
        
        if uploaded_file:
            file_name = uploaded_file.name
            if file_name not in st.session_state.processed_documents:
                with st.spinner(f"Ingesting and indexing `{file_name}`..."):
                    try:
                        if not st.session_state.client:
                            client, embedding_model = setup_qdrant()
                            st.session_state.client = client
                            st.session_state.embedding_model = embedding_model
                        
                        docs = process_document(uploaded_file)
                        if docs:
                            store_embeddings(
                                st.session_state.client,
                                st.session_state.embedding_model,
                                docs,
                                COLLECTION_NAME
                            )
                            st.session_state.processed_documents.append(file_name)
                            st.success(f"✅ Successfully indexed `{file_name}` ({len(docs)} chunks)")
                            st.session_state.setup_complete = True
                    except Exception as e:
                        st.error(f"❌ Ingestion failed: {str(e)}")
            else:
                st.info(f"ℹ️ `{file_name}` is already indexed in the vector database.")

        if st.session_state.processed_documents:
            st.markdown("##### 📚 Indexed Documents")
            for doc in st.session_state.processed_documents:
                st.markdown(f"- 📄 `{doc}`")

    with col2:
        st.subheader("💬 2. Voice Query Interface")
        
        if not st.session_state.setup_complete:
            st.info("👈 Upload and index a document first to activate voice queries.")
        
        query = st.text_input(
            "Ask a question about the documentation:",
            placeholder="e.g., How does the multi-agent pipeline work?",
            disabled=not st.session_state.setup_complete
        )
        
        if query and st.session_state.setup_complete:
            if not st.session_state.openai_api_key:
                st.warning("⚠️ Please provide your OpenAI API Key in the sidebar.")
            else:
                with st.status("Processing query and synthesizing voice...", expanded=True) as status:
                    result = asyncio.run(process_query(
                        query,
                        st.session_state.client,
                        st.session_state.embedding_model,
                        COLLECTION_NAME,
                        st.session_state.openai_api_key,
                        st.session_state.selected_voice
                    ))
                    
                    if result["status"] == "success":
                        status.update(label="✅ Answer & Voice Synthesis Complete!", state="complete")
                        
                        st.markdown("### 📝 Synthesized Answer")
                        st.write(result["text_response"])
                        
                        if "audio_path" in result and os.path.exists(result["audio_path"]):
                            st.markdown(f"### 🔊 Spoken Explanation (`{st.session_state.selected_voice}` voice)")
                            st.audio(result["audio_path"], format="audio/mp3", start_time=0)
                            
                            with open(result["audio_path"], "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                st.download_button(
                                    label="📥 Download Audio (.mp3)",
                                    data=audio_bytes,
                                    file_name=f"voice_rag_response_{st.session_state.selected_voice}.mp3",
                                    mime="audio/mp3"
                                )
                        
                        st.markdown("### 📌 Grounded Sources")
                        for source in result["sources"]:
                            st.markdown(f"- 📄 `{source}`")
                    else:
                        status.update(label="❌ Query Processing Error", state="error")
                        st.error(f"Error: {result.get('error', 'Unknown error occurred')}")


if __name__ == "__main__":
    main()
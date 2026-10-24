# System Architecture & API Overview: Voice RAG Agent

## 1. Introduction
The Voice RAG Agent is an autonomous, multi-agent AI system designed for real-time document intelligence and conversational speech synthesis. By bridging dense vector semantic retrieval (Qdrant + FastEmbed) with OpenAI's dual-agent orchestration framework (Documentation Processor Agent + Audio Script Delivery Agent), the platform converts dense technical documentation into natural, conversational voice explanations.

## 2. Core Architecture Pipeline
1. **Document Ingestion Layer**:
   - Parses multi-page technical PDFs and text files into structural segments using LangChain's RecursiveCharacterTextSplitter (chunk size: 1000 characters, overlap: 200 characters).
   - Generates 384-dimensional dense semantic vectors using FastEmbed's onnx-accelerated BAAI/bge-small-en-v1.5 model.
   - Upserts vectors and metadata payloads (filename, chunk offset, timestamp) into Qdrant vector database.

2. **Semantic Search & Retrieval**:
   - Computes cosine distance similarity over the collection for any incoming user question.
   - Retrieves top-k relevant documentation chunks and constructs a grounded context block.

3. **Dual-Agent Orchestration**:
   - **Agent 1 (Documentation Processor)**: Synthesizes technical truth, formulates concise, citation-grounded answers, and structures points for spoken clarity.
   - **Agent 2 (TTS Script Delivery Specialist)**: Rewrites the technical explanation into conversational phonetic phrasing with strategic pause markers and pronunciation hints.

4. **Streaming Speech Generation**:
   - Streams speech via OpenAI's low-latency TTS models (`gpt-4o-mini-tts` / `tts-1`) across 11 distinct voice personas.
   - Provides local low-latency playback alongside in-browser streaming and downloadable MP3 audio assets.

## 3. Security and Deployment
- Zero raw vector retention on client devices.
- Stateless execution with session state persistence.
- Compatible with local in-memory Qdrant, Dockerized Qdrant, or cloud clusters.

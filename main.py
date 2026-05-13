"""RAG Document Intelligence Engine - Hybrid search + re-ranking over ingested documents."""
  import os, tempfile
  from typing import List
  from fastapi import FastAPI, UploadFile, File
  from fastapi.responses import StreamingResponse
  from pydantic import BaseModel
  from langchain_openai import ChatOpenAI, OpenAIEmbeddings
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
  from langchain_community.retrievers import BM25Retriever
  from langchain.schema import Document
  from dotenv import load_dotenv

  load_dotenv()
  app = FastAPI(title="RAG Document Intelligence Engine", version="1.0.0")
  llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
  embeddings = OpenAIEmbeddings()
  splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
  _documents: List[Document] = []

  class QueryRequest(BaseModel):
      question: str
      top_k: int = 5

  @app.post("/ingest")
  async def ingest_document(file: UploadFile = File(...)):
      suffix = os.path.splitext(file.filename)[1].lower()
      with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
          tmp.write(await file.read()); tmp_path = tmp.name
      loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else Docx2txtLoader(tmp_path)
      chunks = splitter.split_documents(loader.load())
      _documents.extend(chunks); os.unlink(tmp_path)
      return {"ingested_chunks": len(chunks), "total": len(_documents)}

  @app.post("/query")
  async def query(request: QueryRequest):
      if not _documents: return {"error": "No documents ingested yet"}
      retriever = BM25Retriever.from_documents(_documents, k=request.top_k)
      context = "\n\n".join(d.page_content for d in retriever.get_relevant_documents(request.question))
      prompt = f"Answer using only this context:\n\n{context}\n\nQuestion: {request.question}\nAnswer:"
      async def stream():
          async for chunk in llm.astream(prompt): yield chunk.content
      return StreamingResponse(stream(), media_type="text/plain")

  @app.get("/health")
  def health(): return {"status": "online", "docs": len(_documents)}
  
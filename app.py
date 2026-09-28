# import gradio as gr
# import os
# import zipfile
# import shutil
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# UPLOAD_FOLDER = "uploads"
# CHROMA_FOLDER = "chroma_db"

# IGNORED_FOLDERS = {
#     "node_modules",
#     ".git",
#     "venv",
#     "__pycache__",
#     "dist",
#     "build",
#     ".next"
# }

# SUPPORTED_EXTENSIONS = {
#     ".py",
#     ".js",
#     ".jsx",
#     ".ts",
#     ".tsx",
#     ".html",
#     ".css",
#     ".json",
#     ".md",
#     ".txt"
# }

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# def search_code(question):
#     try:
#         if not question:
#             return "🔴 Please enter a question."

#         if not os.path.exists(CHROMA_FOLDER):
#             return "🔴 Please upload and process a project first."

#         vector_store = Chroma(
#             persist_directory=CHROMA_FOLDER,
#             embedding_function=embeddings
#         )

#         results = vector_store.similarity_search(
#             question,
#             k=3
#         )

#         if not results:
#             return "🔴 No relevant code found."

#         response = "## 🔍 Relevant Code\n\n"

#         for i, document in enumerate(results, start=1):
#             source = document.metadata.get(
#                 "source",
#                 "Unknown"
#             )

#             response += f"### Result {i}\n"
#             response += f"📄 **Source:** `{source}`\n\n"
#             response += f"```text\n{document.page_content}\n```\n\n"

#         return response

#     except Exception as e:
#         print("SEARCH ERROR:", repr(e))
#         return f"🔴 **Search Error:** `{repr(e)}`"

# def process_project(zip_file):
#     if zip_file is None:
#         return "🔴 Please upload a ZIP file."

#     if os.path.exists(UPLOAD_FOLDER):
#         shutil.rmtree(UPLOAD_FOLDER)

#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#     with zipfile.ZipFile(zip_file, "r") as zip_ref:
#         zip_ref.extractall(UPLOAD_FOLDER)

#     useful_files = []

#     for root, dirs, files in os.walk(UPLOAD_FOLDER):
#         dirs[:] = [
#             directory
#             for directory in dirs
#             if directory not in IGNORED_FOLDERS
#         ]

#         for file in files:
#             extension = os.path.splitext(file)[1].lower()

#             if extension in SUPPORTED_EXTENSIONS:
#                 file_path = os.path.join(root, file)
#                 useful_files.append(file_path)

#     documents = []

#     for file_path in useful_files:
#         try:
#             with open(file_path, "r", encoding="utf-8") as file:
#                 content = file.read()

#             document = Document(
#                 page_content=content,
#                 metadata={
#                     "source": file_path
#                 }
#             )

#             documents.append(document)

#         except UnicodeDecodeError:
#             continue

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )

#     chunks = text_splitter.split_documents(documents)

#     vectors = embeddings.embed_documents(
#         [chunk.page_content for chunk in chunks]
#     )

#     Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=CHROMA_FOLDER
#     )

#     file_counts = {}

#     for file_path in useful_files:
#         extension = os.path.splitext(file_path)[1].lower()
#         file_counts[extension] = file_counts.get(extension, 0) + 1

#     result = f"""
# 🟢 **Project uploaded successfully!**

# 📁 **Useful files found: {len(useful_files)}**

# 📖 **Files successfully read: {len(documents)}**

# 🧩 **Code chunks created: {len(chunks)}**

# 🧠 **Embeddings created: {len(vectors)}**

# ### 📊 File Types
# """

#     if file_counts:
#         for extension, count in sorted(file_counts.items()):
#             result += f"- `{extension}` : {count} files\n"
#     else:
#         result += "- No supported source files found."

#     return result

# with gr.Blocks(title="CodeBase AI") as demo:
#     gr.Markdown(
#         """
#         # 🧑‍💻 CodeBase AI
#         ### Understand and explore your codebase with AI
#         """
#     )

#     gr.Markdown(
#         "Upload your project ZIP file to get started."
#     )

#     project_file = gr.File(
#         label="📦 Upload Project ZIP",
#         file_types=[".zip"],
#         type="filepath"
#     )

#     process_button = gr.Button(
#         "⚙️ Process Project",
#         variant="primary"
#     )

#     status = gr.Markdown(
#         "🔴 No project uploaded"
#     )

#     process_button.click(
#         fn=process_project,
#         inputs=project_file,
#         outputs=status
#     )

#     gr.Markdown("---")

#     gr.Markdown(
#         "## 🔍 Ask Questions About Your Code"
#     )

#     question = gr.Textbox(
#         label="Ask your codebase",
#         placeholder="Example: Where is authentication implemented?"
#     )

#     search_button = gr.Button(
#         "🔎 Search Code",
#         variant="primary"
#     )

#     search_result = gr.Markdown()

#     search_button.click(
#         fn=search_code,
#         inputs=question,
#         outputs=search_result
#     )

# demo.launch() 

import os 
from dotenv import load_dotenv
import serpapi
load_dotenv()

from langchain.agents import create_agent 
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.tools import tool
from serpapi import GoogleSearch

api_key=os.getenv("GOOGLE_API_KEY")
tavily_api_key=os.getenv("TAVILY_API_KEY")
serp_api_key=os.getenv("SERPAPI_KEY")



# Step 1: Initialize the Model


model = init_chat_model(
    "google_genai:gemini-3.5-flash-lite",
    api_key=api_key

)

# model=init_chat_model(
#     "google_genai:gemini-2.5-pro",
#     api_key=api_key
# )



# Step 2: Create Destination Research Tool (Tavily)
tavily_api_key=os.getenv("TAVILY_API_KEY")
destination_research_tool =TavilySearch(
    max_results=5,
    search_depth="advanced",
    tavily_api_key=tavily_api_key

)


# Step 3: Create Flight Search Tool
@tool
def search_flights(origin: str, destination: str, date: str) -> list:
    """Search for flights between two locations on a specified date."""

    params = {
        "engine": "google_flights",
        "departure_id": origin,
        "arrival_id": destination,
        "outbound_date": date,
        "api_key": serp_api_key,
        "hl": "en",
        "gl": "in"
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    return results

# Step 4: Define System Prompt
system_prompt="""
You are a TravelBuddy assistant that helps travelers plan trips.
You have access to these tools:
- destination_research_tool: Research attractions, culture, and travel tips
- search_flights: Find flight options (use IATA codes: HYD=Hyderabad, GOI=Goa, BOM=Mumbai, DEL=Delhi, BLR=Bangalore)
Help the traveler by researching destinations and finding flights.
Present results in a clean, readable format with clear sections.
Don't use markdown format.
"""

agent=create_agent(
    model=model,
    tools=[destination_research_tool,search_flights],
    system_prompt=system_prompt
)



# Step 5: Create and Run the Agent
user_query="I want to visit Goa from Hyderabad on 2026-12-15. Show me top attractions and flight options"
mesagges=[{"role":"user","content":user_query}]
response=agent.invoke({"messages":mesagges})

print(response)


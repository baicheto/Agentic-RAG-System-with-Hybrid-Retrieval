import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from duckduckgo_search import DDGS

load_dotenv()


llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=1000,
    max_retries=2,
)

crew_llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=500,
    temperature=0.7,
    tool_choice="none"
)


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Search the web using DuckDuckGo, no API key required."

    def _run(self, query: str) -> str:
        for attempt in range(3):  # retry up to 3 times
            try:
                with DDGS() as ddgs:
                    results = ddgs.text(query, max_results=5)
                    if results:
                        return "\n".join(
                            f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}"
                            for r in results
                        )
            except Exception as e:
                print(f"[DuckDuckGo] Attempt {attempt+1} failed: {e}")
        return "No results found."
    

# Vector DB Setup
###################################
def setup_vector_db(pdf_path: str):
    """
    Loads a PDF file, split it into overlapping text chunks,
    embed them using a HuggingFace sentence transformer,
    and store everything in a FAISS vector index for fast similarity search.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
 
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db


# Local Retrieval Functions
###################################
def retrieve_local_docs(vector_db, query: str, k: int = 5):
    """
    Retrieve the top-k most semantically similar document chunks
    from the local FAISS index for a given query.
    """
    docs = vector_db.similarity_search(query, k=k)
    return docs
 
 
def format_docs(docs):
    """
    Format a list of retrieved document chunks into a readable string,
    including the page number metadata for traceability.
    """
    formatted = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page", "unknown")
        text = doc.page_content.strip()
        formatted.append(f"[Local Doc {i} | Page {page}]\n{text}")
    return "\n\n".join(formatted)


# Local Context Summary Builder
#####################################
def build_local_context_summary(vector_db) -> str:
    """
    Build a richer summary of the local document's content
    by querying multiple representative topics and combining results.
    This gives the router a more complete picture of what the PDF covers.
    """
    probe_queries = [
        "main topics and key concepts",
        "introduction and overview",
        "methodology and approach",
        "conclusions and findings",
    ]
 
    seen_pages = set()
    all_chunks = []
 
    for probe in probe_queries:
        docs = retrieve_local_docs(vector_db, probe, k=3)
        for doc in docs:
            page = doc.metadata.get("page", "unknown")
            # Deduplicate by page number to avoid repetition
            if page not in seen_pages:
                seen_pages.add(page)
                all_chunks.append(doc)
 
    return format_docs(all_chunks)


# Routing Function
###############################
def route_query(query: str, local_context_summary: str) -> str:
    """
    Use the LLM to decide the best retrieval strategy for a given query:
    - 'local': answer can be found in the local documents
    - 'web': answer requires live or up-to-date web information
    - 'both': answer benefits from combining both sources
 
    Falls back to 'both' if the LLM returns an unexpected response.
    """
    messages = [
        (
            "system",
            "You are a routing assistant for a RAG system. "
            "Decide whether the user's question should be answered using LOCAL documents, WEB search, or BOTH. "
            "Return only one word: local, web, or both."
        ),
        (
            "system",
            f"Summary of local document knowledge:\n{local_context_summary[:4000]}"  
        ),
        ("human", query),
    ]
 
    response = llm.invoke(messages).content.strip().lower()
 
    if response not in {"local", "web", "both"}:
        return "both"
    return response


# Document Relevance Grader
###############################
def grade_local_docs(query: str, docs) -> str:
    """
    Ask the LLM to judge whether the retrieved local documents
    are relevant enough to answer the given question.
    Returns 'relevant' or 'not_relevant'.
    """
    doc_text = format_docs(docs)
 
    messages = [
        (
            "system",
            "You are a document relevance grader. "
            "Decide whether the retrieved documents are relevant enough to answer the user's question. "
            "Return only one label: relevant or not_relevant."
        ),
        ("human", f"Question:\n{query}\n\nRetrieved Documents:\n{doc_text[:5000]}")
    ]
 
    response = llm.invoke(messages).content.strip().lower()
 
    if response not in {"relevant", "not_relevant"}:
        return "not_relevant"
    return response


# Query Rewriter
###############################
def rewrite_query(query: str) -> str:
    """
    Use the LLM to rephrase the user's question into a form
    that is more suitable for semantic retrieval in a vector database.
    Falls back to the original query if rewriting fails.
    """
    messages = [
        (
            "system",
            "You are a query rewriting assistant. "
            "Rewrite the user's question so it is clearer and better for semantic retrieval. "
            "Preserve the meaning. Return only the rewritten query."
        ),
        ("human", query),
    ]
 
    response = llm.invoke(messages).content.strip()
    return response if response else query


# Web Agents
#################################
def setup_web_scraping_crew() -> Crew:
    """
    Create and return a CrewAI crew composed of two agents:
    1. A web search agent that finds the best URL for the query.
    2. A web scraper agent that extracts and summarizes the page content.
 
    The crew is intended to be instantiated once and reused across calls.
    """
    search_tool = DuckDuckGoSearchTool()
    scrape_website_tool = ScrapeWebsiteTool()
 
    web_search_agent = Agent(
        role="Expert Web Search Agent",
        goal="Find the most relevant web source for the user's query",
        backstory="You are highly skilled at locating useful, trustworthy web sources.",
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )
 
    web_scraper_agent = Agent(
        role="Expert Web Scraper Agent",
        goal="Extract and summarize the most relevant content from the chosen web page",
        backstory="You are highly skilled at extracting useful information from web pages accurately.",
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )
 
    search_task = Task(
        description=(
            "Search the web for the most relevant source about the topic: '{topic}'. "
            "Find one strong source and provide the best link."
        ),
        expected_output=(
            "The best web source for '{topic}', including the URL and a short reason why it is relevant."
        ),
        tools=[search_tool],
        agent=web_search_agent,
    )
 
    scraping_task = Task(
        description=(
            "Use the selected web source for the topic '{topic}'. "
            "Extract and summarize the most important content from the page clearly."
        ),
        expected_output=(
            "A concise but informative summary of the web page content for '{topic}'."
        ),
        tools=[scrape_website_tool],
        agent=web_scraper_agent,
    )
 
    crew = Crew(
        agents=[web_search_agent, web_scraper_agent],
        tasks=[search_task, scraping_task],
        verbose=1,
        memory=False,
    )
 
    return crew
 
 
def get_web_content(query: str, crew: Crew) -> str:
    try:
        result = crew.kickoff(inputs={"topic": query})
        return result.raw if hasattr(result, "raw") else str(result)
    except Exception as e:
        # Log the error and return an empty string so the pipeline can continue
        print(f"[Web Agent ERROR] Web retrieval failed: {e}")
        return ""
    

# Final Answer Generator
###############################
def generate_final_answer(query: str, context: str) -> str:
    """
    Generates a grounded final answer using the LLM, based strictly
    on the provided context (local docs + web content).
    The model is instructed to clearly state the source of its answer.
    """
    messages = [
        (
            "system",
            "You are a helpful assistant. Answer the user's question using only the provided context. "
            "If the context is insufficient, say so clearly. "
            "When possible, mention whether the answer came from local documents or web findings."
        ),
        ("system", f"Context:\n{context[:8000]}"),  
        ("human", query),
    ]
 
    response = llm.invoke(messages)
    return response.content
 


# Answer Grounding Verifier
###############################
def verify_answer_grounding(query: str, context: str, answer: str) -> str:
    """
    Asks the LLM to verify whether the generated answer is actually
    supported by the provided context, or if it contains hallucinations.
    Returns 'grounded' or 'ungrounded'.
    """
    messages = [
        (
            "system",
            "You are a grounding checker. "
            "Decide whether the answer is fully supported by the provided context. "
            "Return only one label: grounded or ungrounded."
        ),
        (
            "human",
            f"Question:\n{query}\n\nContext:\n{context[:8000]}\n\nAnswer:\n{answer}"  
        ),
    ]
 
    response = llm.invoke(messages).content.strip().lower()
 
    if response not in {"grounded", "ungrounded"}:
        return "ungrounded"
    return response


# Agentic RAG Pipeline
################################
def process_query_agentic(query: str, vector_db, local_context_summary: str, web_crew: Crew):
    """
    Full agentic RAG pipeline:
 
    1.  Route the query -> local / web / both
    2.  Retrieve local docs if route includes local
    3.  Grade local doc relevance
    4.  If not relevant: rewrite query and retry (up to MAX_REWRITES times)
    5.  Fall back to web if local is still not relevant
    6.  Retrieve web content if route includes web or local failed
    7.  Merge local + web context
    8.  Generate final answer
    9.  Verify grounding; if ungrounded, flag the answer with a warning instead of discarding it entirely
    """
 
    MAX_REWRITES = 3  
 
    state = {
        "original_query": query,
        "current_query": query,
        "route": None,
        "actual_sources_used": [],   
        "local_docs": [],
        "local_relevance": None,
        "rewrites": 0,
        "web_content": "",
        "used_web": False,
        "final_context": "",
        "answer": "",
        "grounding": None,
        "grounding_warning": False, 
    }
 
    print(f"\n{'='*50}")
    print(f"Processing query: {query}")
    print(f"{'='*50}")
 

    
    # 1: Route the query
    route = route_query(query, local_context_summary)
    state["route"] = route
    print(f"[Router] Decision: {route}")

   
    # 2 & 3: Local retrieval + relevance grading with multi-attempt rewriting
    if route in {"local", "both"}:
        docs = retrieve_local_docs(vector_db, state["current_query"], k=5)
        state["local_docs"] = docs
    
        # grade using the current query (not always the original)
        relevance = grade_local_docs(state["current_query"], docs)
        state["local_relevance"] = relevance
        print(f"[Local Retrieval] Relevance: {relevance}")
    
        # retry rewriting up to MAX_REWRITES times 
        while relevance == "not_relevant" and state["rewrites"] < MAX_REWRITES:
            rewritten = rewrite_query(state["current_query"])
            state["current_query"] = rewritten
            state["rewrites"] += 1
            print(f"[Query Rewriter] Attempt {state['rewrites']}: {rewritten}")
    
            docs = retrieve_local_docs(vector_db, state["current_query"], k=5)
            state["local_docs"] = docs
    
            # always grade with the latest rewritten query
            relevance = grade_local_docs(state["current_query"], docs)
            state["local_relevance"] = relevance
            print(f"[Local Retrieval After Rewrite #{state['rewrites']}] Relevance: {relevance}")
    
        if relevance == "relevant":
            state["actual_sources_used"].append("local")


    # 4: Web retrieval if needed
    should_use_web = (
        route == "web"
        or route == "both"
        or state["local_relevance"] == "not_relevant"  # local failed even after rewrites
        )
    
    if should_use_web:
        print("[Web] Retrieving web content...")
        # pass the pre-built crew; handle failure gracefully inside get_web_content()
        web_content = get_web_content(query, web_crew)
        state["web_content"] = web_content
        state["used_web"] = True
    
        if web_content:
            state["actual_sources_used"].append("web")
            print("[Web] Content retrieved successfully")
        else:
            print("[Web] No content retrieved (agent may have failed)")
    
        # update route label to reflect actual behavior
        if route == "local" and should_use_web:
            state["route"] = "local -> web_fallback"

    
    # 5: Build merged context from all sources
    
    local_context = format_docs(state["local_docs"]) if state["local_docs"] else ""
    web_context = f"[Web Findings]\n{state['web_content']}" if state["web_content"] else ""
    
    final_context_parts = []
    if local_context:
        final_context_parts.append(local_context)
    if web_context:
        final_context_parts.append(web_context)
    
    state["final_context"] = "\n\n".join(final_context_parts)
    
    # Edge case: no context at all
    if not state["final_context"].strip():
        state["answer"] = (
            "No relevant information was found in local documents or via web search. "
            "Please try a more specific question or provide additional source material."
        )
        state["grounding"] = "n/a"
        return state


    # 6: Generate the final answer based on the merged context
    answer = generate_final_answer(query, state["final_context"])
    state["answer"] = answer
    print("[Answer Generator] Answer created")

    
    # 7: Verify grounding
    grounding = verify_answer_grounding(query, state["final_context"], answer)
    state["grounding"] = grounding
    print(f"[Grounding Check] {grounding}")
    
    if grounding == "ungrounded":
        state["grounding_warning"] = True
        # keep the answer but warn the user
        state["answer"] = (
            f"{answer}\n\n"
            "Warning: This answer could not be fully verified against the retrieved context. "
            "Please treat it with caution and consider refining your question."
        )
    return state



def main():
    pdf_path = "Optimization-of-Dynamic-Pricing-Models-for-Consumer-Segmentation-Markets-and-Analysis-of-Big-Data-Driven-Marketing-Strategies.pdf"
 
    print("Setting up vector database...")
    vector_db = setup_vector_db(pdf_path)
 
    # use the improved multi-probe summary builder for a richer routing signal
    print("Building local context summary for router...")
    local_context_summary = build_local_context_summary(vector_db)
 
    # build the web crew once here and reuse it across all queries
    print("Setting up web scraping crew...")
    web_crew = setup_web_scraping_crew()
 
    query = "What is Keyword Extraction?"
    result = process_query_agentic(query, vector_db, local_context_summary, web_crew)
 
    print("\n========== FINAL RESULT ==========")
    print(f"Original Query  : {result['original_query']}")
    print(f"Route           : {result['route']}")
    print(f"Sources Used    : {result['actual_sources_used']}")
    print(f"Rewrites        : {result['rewrites']}")
    print(f"Used Web        : {result['used_web']}")
    print(f"Grounding       : {result['grounding']}")
    print(f"Grounding Warn  : {result['grounding_warning']}")
    print("\nAnswer:")
    print(result["answer"])
 
 
if __name__ == "__main__":
    main()

    



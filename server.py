import os 
import yfinance as yf
import chromadb
from datetime import datetime
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from crewai import Agent, Task, Crew, Process, LLM

# 1. Initialize MCP
mcp = FastMCP("AlphaMind_Server")

# 2. Setup Local LLM (Use the ollama/ prefix for stability)
# Ensure you have run 'ollama pull llama3' in your terminal
local_llm = LLM(
    model="ollama/llama3",
    base_url="http://localhost:11434"
)

# --- RAG SYSTEM ---
class FinancialRAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.PersistentClient(path="./financial_db")
        self._initialize_kb()

    def _initialize_kb(self):
        collection = self.chroma_client.get_or_create_collection(name="financial_knowledge")
        if collection.count() == 0:
            docs = [
                "BULL MARKET: PERIOD OF SUSTAINED STOCK PRICE INCREASES. STRATEGY: GROWTH STOCKS.",
                "BEAR MARKET: PERIOD OF SUSTAINED DECLINES. STRATEGY: DEFENSIVE STOCKS.",
                "TECHNICAL INDICATORS: RSI (MOMENTUM), MACD (TREND), BOLLINGER BANDS (VOLATILITY)."
            ]
            collection.add(documents=docs, ids=["doc1", "doc2", "doc3"])

    def get_context(self, query):
        collection = self.chroma_client.get_collection("financial_knowledge")
        results = collection.query(query_texts=[query], n_results=2)
        return "\n".join(results['documents'][0])

rag = FinancialRAGSystem()

# --- CORE LOGIC (Streamlit will call THESE) ---

def get_market_data_logic(symbol: str) -> str:
    """Plain Python function for market data."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        price = info.get('currentPrice', 'N/A')
        sector = info.get('sector', 'N/A')
        return f"Symbol: {symbol}, Price: {price}, Sector: {sector}"
    except Exception as e:
        return f"Error: {str(e)}"

def agent_report_logic(symbol: str, user_query: str) -> str:
    """Plain Python function for Agent Analysis."""
    context = rag.get_context(user_query)
    market_info = get_market_data_logic(symbol)

    researcher = Agent(
        role='Senior Financial Analyst',
        goal=f'Analyze {symbol} given this context: {context}',
        backstory='Expert at synthesizing RAG data and market trends.',
        llm=local_llm,
        verbose=True,
        allow_delegation=False
    )

    analysis_task = Task(
        description=f"Market Info: {market_info}. User Question: {user_query}",
        expected_output="A professional financial summary with a recommendation.",
        agent=researcher
    )

    crew = Crew(agents=[researcher], tasks=[analysis_task], process=Process.sequential, memory=False)
    return str(crew.kickoff())

# --- MCP TOOLS (MCP Server will use THESE) ---

@mcp.tool()
def get_market_data(symbol: str) -> str:
    """MCP Tool wrapper for market data."""
    return get_market_data_logic(symbol)

@mcp.tool()
def agent_financial_report(symbol: str, user_query: str) -> str:
    """MCP Tool wrapper for agent report."""
    return agent_report_logic(symbol, user_query)

if __name__ == "__main__":
    mcp.run()

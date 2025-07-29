from utils.rag_utils import load_and_store_pdf, query_context
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import litellm
# from utils import rag_utils   # ✅ NEW
from utils.rag_utils import VECTOR_DB_DIR
# Load environment variables
load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# litellm.api_key = os.environ["GROQ_API_KEY"]
# llm_model = "groq/llama3-8b-8192"

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
litellm.api_key = os.environ["OPENAI_API_KEY"]
llm_model = "openai/gpt-3.5-turbo"  # or "openai/gpt-4o"
# Then use llm_model for your agents as before

# Default directory where RAG vector store is saved
RAG_DIR = VECTOR_DB_DIR

# Define agents (same as before)
researcher_agent = Agent(
    role='Researcher',
    goal='Collect background data, statistics, and funding body information.',
    backstory='Expert in grant funding research, analyzes relevant data and trends.',
    llm=llm_model
)

idea_agent = Agent(
    role='Idea Extractor',
    goal='Summarize the user input and define the grant problem statement.',
    backstory='Understands grant context and structures the foundational idea.',
    llm=llm_model
)

writer_agent = Agent(
    role='Proposal Writer',
    goal='Generate a compelling, detailed grant proposal.',
    backstory='Specializes in persuasive proposal writing and storytelling.',
    llm=llm_model
)

formatter_agent = Agent(
    role='Formatter',
    goal='Structure the content into standard grant proposal format.',
    backstory='Ensures consistent formatting, clear sections, and readability.',
    llm=llm_model
)

proofreader_agent = Agent(
    role='Proofreader',
    goal='Polish grammar, tone, and check compliance with grant norms.',
    backstory='Refines language and ensures professionalism.',
    llm=llm_model
)

# ✅ MAIN: RAG-INTEGRATED CREW


def build_crew(grant_topic: str, pdf_path: str = None):
    # If user uploaded a new PDF, store it
    if pdf_path:
        load_and_store_pdf(pdf_path)
        rag_context = query_context(grant_topic, k=3)
    else:
        rag_context = ""
    # try:
    #     rag_context = load_rag_context(RAG_DIR, grant_topic)
    # except Exception as e:
    #     print(f"⚠️ RAG loading failed: {e}")

    task1 = Task(
        description=f"Research background info and statistics for the topic: {grant_topic}. "
        f"Use this context if relevant:\n{rag_context}",
        expected_output="Concise research summary (facts, programs, numbers, trends).",
        agent=researcher_agent
    )

    task2 = Task(
        description="Extract the core idea and problem statement from the user input and research. "
                    f"Use this context if helpful:\n{rag_context}",
        expected_output="List of objectives, goals, and target audience.",
        agent=idea_agent
    )

    task3 = Task(
        description="Write a full draft of the grant proposal based on previous findings.",
        expected_output="Drafted proposal with introduction, goals, methodology, impact.",
        agent=writer_agent
    )

    task4 = Task(
        description="Format the draft into a formal grant proposal structure.",
        expected_output="Formatted text with clear headers: Executive Summary, Problem, Solution, Budget.",
        agent=formatter_agent
    )

    task5 = Task(
        description="Proofread and refine the entire proposal for clarity and tone.",
        expected_output="Final, polished grant proposal document.",
        agent=proofreader_agent
    )

    crew = Crew(
        agents=[researcher_agent, idea_agent, writer_agent,
                formatter_agent, proofreader_agent],
        tasks=[task1, task2, task3, task4, task5],
        verbose=True
    )

    return crew

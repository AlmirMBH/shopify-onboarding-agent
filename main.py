import os
import getpass
import sys
import subprocess
import subprocess
import sys

# Set the Environment Variables
os.environ['USER_AGENT'] = 'myagent'
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter LangSmith API key: ")

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

os.environ["HF_TOKEN"] = getpass.getpass("Enter Huggingface token: ")

if not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = getpass.getpass("Enter API key for MistralAI: ")

# import requests
# headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}
# response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)
# if response.status_code == 200:
#     print("✅ Hugging Face token is valid.")
# else:
#     print("❌ Hugging Face token is invalid or unauthorized.")



# Install Packages
def install(*packages):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def validate_and_install(packages):
    for package in packages:
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} not found. Installing...")
            install(package)

packages_to_validate = [
    "langchain_groq",
    "langchain_mistralai",
    "langchain_community",
    "beautifulsoup4",
    "langgraph",
    "pandas"
]

validate_and_install(packages_to_validate)
import bs4

# Import the Required Libraries
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

llm = init_chat_model("llama3-8b-8192", model_provider="groq")
embeddings = MistralAIEmbeddings(model="mistral-embed")
vector_store = InMemoryVectorStore(embeddings)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",), 
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))), 
)

# from langchain_community.document_loaders import GoogleDocsLoader
# DOC_ID = "YOUR_GOOGLE_DOC_ID"
# loader = GoogleDocsLoader(document_ids=[DOC_ID])

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
_ = vector_store.add_documents(documents=all_splits)
graph_builder = StateGraph(MessagesState)


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2) # k is the number of search results
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState):
    system_message = SystemMessage( # Append a clear system instruction
        "You are an assistant. Only use the 'retrieve' tool for domain-specific questions. "
        "For general or conversational queries such as greetings or 'can you help me with something', "
        "provide a direct answer without calling any tools."
    )

    messages = [system_message] + state["messages"]
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(messages)

    # Check if the response includes an empty tool-use message
    tool_calls = getattr(response, "tool_calls", None)

    # If tool_calls attribute exists but is empty, assume no tool was needed.
    if tool_calls is not None and not tool_calls:
        fallback = {"role": "ai", "content": "Hello! How can I assist you today?"}
        return {"messages": [fallback]}
    return {"messages": [response]}


tools = ToolNode([retrieve])


def generate(state: MessagesState):
    # Get generated ToolMessages
    recent_tool_messages = []

    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    # docs_content = "\n\n".join(tool_messages.content for tool_message in tool_messages)
    docs_content = "\n\n".join(tool_message.content for tool_message in tool_messages if hasattr(tool_message, 'content'))

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )

    conversation_messages = [
        message
        for message in state["messages"] if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]

    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # This is the moment when we start the generation, see the graph in the sections below
    response = llm.invoke(prompt)
    return {"messages": [response]}


graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)
graph_builder.set_entry_point("query_or_respond")

graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)

graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver() # Init chat persistance
graph = graph_builder.compile(checkpointer=memory) # Compile with the peristance layer (similar to git logs)
config = {"configurable": {"thread_id": "test_123"}} # Specify an ID for the thread (similar to git branch)

if __name__ == "__main__":
    

    # TODO: Create a simple prompt window
    # input_message = "Mr. Chat, how do you do!"
    # for step in graph.stream(
    #     {"messages": [{"role": "user", "content": input_message}]},
    #     stream_mode="values",
    #     config=config
    # ):
    #     step["messages"][-1].pretty_print()

    input_message = "What is Task Decomposition?"
    final_response = None

    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config
    ):
        last_message = step["messages"][-1].content
        if last_message:
            final_response = last_message

    print("\nRESPONSE:", final_response if final_response else "No response")
        
    # input_message = "Can you look up some common ways of doing it?"

    # for step in graph.stream(
    #     {"messages": [{"role": "user", "content": input_message}]},
    #     stream_mode="values",
    #     config=config,
    # ):
    #     step["messages"][-1].pretty_print()
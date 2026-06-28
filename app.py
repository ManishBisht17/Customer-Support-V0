import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from rag_tool import search_knowledge_base
from memory_tool import remember_fact, recall_facts   # NEW

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [search_knowledge_base, remember_fact, recall_facts]   # NEW tools added
llm_with_tools = llm.bind_tools(tools)

tools_map = {
    "search_knowledge_base": search_knowledge_base,
    "remember_fact": remember_fact,
    "recall_facts": recall_facts,
}

messages = [SystemMessage(
    "You are a helpful customer support agent. "
    "Always search the knowledge base before answering questions about "
    "shipping, returns, refunds, or account issues. "
    "If the customer mentions a problem worth remembering (like a complaint "
    "or damaged order), use remember_fact to save it. "
    "If you want to check the customer's history, use recall_facts. "
    "Be concise and friendly."
)]

def run_agent(query: str, customer_id: str = "test_customer_1") -> str:
    # we add the customer_id into the message so the LLM knows who it's talking to
    messages.append(HumanMessage(f"[customer_id: {customer_id}] {query}"))

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            print(f"\n[Using tool: {tool_call['name']} with {tool_call['args']}]")
            result = tools_map[tool_call["name"]].invoke(tool_call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

if __name__ == "__main__":
    print("Support Agent ready! Type 'exit' to quit.\n")
    while True:
        user_input = input("Customer: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = run_agent(user_input)
        print(f"\nAgent: {response}\n")
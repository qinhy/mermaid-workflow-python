import os
import json
import asyncio

import aiohttp
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

from MermaidWorkflowEngine import MermaidWorkflowEngine

SERVER_SCRIPT = "./9.0.lib_with_mcp_server.py"
async def get_tools()->str:
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)
    async with Client(transport) as client:
        tools = await client.list_tools()
        return json.dumps([tool.model_dump() for tool in tools])

async def call_tool(tool_name, tool_args)->str:
    transport = PythonStdioTransport(script_path=SERVER_SCRIPT)
    async with Client(transport) as client:
        result = await client.call_tool(tool_name, tool_args)
        return [r.text for r in result][0]

def mcp_to_openai_tool(tools) -> list:
    if type(tools) == str:
        tools = json.loads(tools)
    if not isinstance(tools, list):
        tools = [tools]        
    openai_tools = []
    for tool in tools:
        tool = json.loads(json.dumps(tool))
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool['name'],
                "description": tool['description'],
                "parameters": tool['inputSchema'],
            },
        }
        openai_tools.append(openai_tool)        
    return openai_tools

async def llm(messages, tools=[]):
    API_KEY = os.getenv("OPENAI_API_KEY")
    API_URL = "https://api.openai.com/v1/chat/completions"
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    if type(messages) == str:
        messages = [{"role": "user", "content": messages}]
    body = {
        "model": "gpt-4.1", "messages": messages,
        "tools": tools, "tool_choice": "auto"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL,
                    headers=HEADERS, json=body) as response:
            response.raise_for_status()
            result = await response.json()
            message = result["choices"][0]["message"]
            # Return full message including potential tool calls
            if "tool_calls" in message:
                return {
                    "content": message.get("content", ""),
                    "tool_calls": message["tool_calls"],
                }
            return message["content"]
# -------- Main --------
if __name__ == "__main__":
    mcp_ts = asyncio.run(get_tools())
    tsd = {t['name']:t for t in json.loads(mcp_ts)}
    # print(asyncio.run(call_tool('Start',{'para': {'x': 1, 'y': 2, 'z': 3}}
                                # )))
    engine = MermaidWorkflowEngine(model_registry = tsd)
    def run_tool(tool_name, tool_args):
        if type(tool_args) is not str:
            tool_name = tool_name.__class__.__name__
        res_json = asyncio.run(call_tool(tool_name, tool_args))
        res = json.loads(res_json)
        return res

    test_graph = """
graph TD
    Start["{'para': {'x': 10, 'y': 5, 'z': 3}}"]
    Multiply["{'para': {'factor': 4}}"]

    Start -- "{'x':'a','y':'b'}" --> Add
    Start -- "{'y':'b'}" --> Subtract
    Add -- "{'sum':'a'}" --> Subtract
    Subtract -- "{'result':'x'}" --> Multiply
    Multiply --> ValidateResult
    ValidateResult --> End
"""
    # results = engine.run(test_graph,run_tool)

    # print("🔍 Setting available tools...")
    # op_ts = mcp_to_openai_tool(mcp_ts)
    # print("#### 🤖 Asking LLM about available tools...")
    # response = asyncio.run(llm('Tell me your available tools.',tools=op_ts))
    # print(f"🔹 Response:\n{response}\n")
    
    print("#### 🤖 Asking LLM to generate a new graph...")
    prompt = f'''
**I have an example graph and a list of all available tools in JSON format in the following code blocks.
Please create a new graph example for me.**

```json
{json.dumps(tsd,indent=4)}
```

```mermaid
{test_graph}
```

### 📌 Mermaid Graph Protocol (for beginners):

* `graph TD` → Start of a top-down Mermaid flowchart
* `Node_Name["{{...}}"]` → Define a node with initialization parameters (in JSON-like format)
* `A --> B` → Connect node A to node B (no field mapping)
* `A -- "{{'x':'y'}}" --> B` → Map output `'x'` from A to input `'y'` of B
* Use **valid field names** from each tool's input/output schema
* Always **end with** a final node like: `C -- "{{'valid':'valid'}}" --> End`
'''
    
    response = asyncio.run(llm(prompt))
    print(f"🔹 Response:\n{response}\n")
    print(f"🔹 Graph:\n{engine.extract_mermaid_text(response)}\n")
    print(engine.run(engine.extract_mermaid_text(response),run_tool))
import os
import json
import asyncio

import aiohttp
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

from MermaidWorkflowEngine import MermaidWorkflowEngine

SERVER_SCRIPT = "./ex_1_1_vison_mcp_server.py"
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

async def llm(messages, instructions=None, tools=[]):
    API_KEY = os.getenv("OPENAI_API_KEY")
    API_URL = "https://api.openai.com/v1/responses"
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    if type(messages) == str:
        messages = [{"role": "user", "content": messages}]
    body = {
        "model": "o4-mini", "input": messages,"instructions":instructions,
        "reasoning": {"effort": "medium","summary": "auto"},
        "tools": tools, "tool_choice": "auto"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL,
                    headers=HEADERS, json=body) as response:
            result = await response.json()
            print(result)
            response.raise_for_status()
            message = {"content":""}
            # message = result["choices"][0]["message"]
            if result['output'][0]['summary']:
                message['content'] += '\n'.join([i['text'] for i in result['output'][0]['summary']])+"\n"
            message = {"content":result['output'][1]['content'][0]['text']}
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
    mcp_ts = json.loads(mcp_ts)
    tsd = {t['name']:t for t in mcp_ts}

    engine = MermaidWorkflowEngine(model_registry = tsd)
    def run_tool(tool_name, tool_inputs):
        tool_inputs = {k:(v.model_dump() if hasattr(v,'model_dump') else v) for k,v in tool_inputs.items()}
        if type(tool_name) is not str:
            tool_name = tool_name.__class__.__name__
        # print(f"🔹 Calling tool: {tool_name}")
        # print(f"🔹 inputs: {tool_inputs}")
        res_json = asyncio.run(call_tool(tool_name, tool_inputs))
        res = json.loads(res_json)
        return res
    
    test_graph = """
graph TD
    LoadImage["{'para': {'path': './tmp/input.jpg'}}"]
    ResizeImage["{'para': {'width': 512, 'height': 512}}"]
    CropImage["{'para': {'left': 50, 'upper': 50, 'right': 462, 'lower': 462}}"]
    BlurImage["{'para': {'radius': 3}}"]
    RotateImage["{'para': {'angle': 270}}"]
    AdjustImage["{'para': {'brightness': 1.2, 'contrast': 1.3, 'saturation': 0.9}}"]
    FlipImage["{'para': {'mode': 'vertical'}}"]
    WatermarkImage["{'para': {'text':'CONFIDENTIAL', 'position':'bottom_right', 'opacity':0.5}}"]
    FilterImage["{'para': {'filter_type': 'sharpen'}}"]
    ConvertImageFormat["{'para': {'format': 'png', 'quality':90}}"]

    LoadImage -- "{'path':'path'}" --> ResizeImage
    ResizeImage -- "{'path':'path'}" --> CropImage
    CropImage -- "{'path':'path'}" --> BlurImage
    BlurImage -- "{'path':'path'}" --> GrayscaleImage
    GrayscaleImage -- "{'path':'path'}" --> RotateImage

    RotateImage -- "{'path':'path'}" --> AdjustImage
    AdjustImage -- "{'path':'path'}" --> FlipImage
    FlipImage -- "{'path':'path'}" --> WatermarkImage
    WatermarkImage -- "{'path':'path'}" --> FilterImage
    FilterImage -- "{'path':'path'}" --> ConvertImageFormat
"""
    # results = engine.run(test_graph,run_tool)

    # print("🔍 Setting available tools...")
    # op_ts = mcp_to_openai_tool(mcp_ts)

    # print("#### 🤖 Asking LLM about available tools...")
    # response = asyncio.run(llm('Tell me your available tools.',tools=op_ts))
    # print(f"🔹 Response:\n{response}\n")
    
    print("#### 🤖 Asking LLM to generate a new graph...")
    prompt = f'''
**I have an example graph and a list of all available tools in JSON format in the following code blocks.**

```json
{json.dumps(tsd,indent=4)}
```

```mermaid
{test_graph}
```
{engine.protocol}
'''    
    response = asyncio.run(llm("Please create a new simple graph. Use image of ./tmp/input.jpg",
                            instructions=prompt))
    print(f"🔹 Response:\n{response}\n")
    print(f"🔹 Graph:\n{engine.extract_mermaid_text(response)}\n")
    print(engine.run(engine.extract_mermaid_text(response),run_tool))


























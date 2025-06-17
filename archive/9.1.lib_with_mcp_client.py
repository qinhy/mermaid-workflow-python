import os
import json
import asyncio

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

# -------- Main --------
if __name__ == "__main__":
    ts = asyncio.run(get_tools())
    tsd = {t['name']:t for t in json.loads(ts)}
    # print(asyncio.run(call_tool('Start',{'para': {'x': 1, 'y': 2, 'z': 3}}
                                # )))
    # print(tsd)
    engine = MermaidWorkflowEngine(model_registry = tsd)
    def run_tool(tool_name, tool_args):
        if type(tool_args) is not str:
            tool_name = tool_name.__class__.__name__
        res_json = asyncio.run(call_tool(tool_name, tool_args))
        res = json.loads(res_json)
        return res

    results = engine.run("""
graph TD
    Start["{'para': {'x': 10, 'y': 5, 'z': 3}}"]
    Multiply["{'para': {'factor': 4}}"]

    Start -- "{'x':'a','y':'b'}" --> Add
    Start -- "{'y':'b'}" --> Subtract
    Add -- "{'sum':'a'}" --> Subtract
    Subtract -- "{'result':'x'}" --> Multiply
    Multiply --> ValidateResult
    ValidateResult --> End
""",run_tool)
    
#     results = engine.run("""
# graph TD
#     Start["{'para': {'x': 20, 'y': 6, 'z': 2}}"]

#     Start -- "{'x':'numerator','y':'denominator'}" --> Divide
#     Start -- "{'x':'a','z':'b'}" --> Modulus
#     Divide -- "{'quotient':'product'}" --> ValidateResult
#     ValidateResult --> End
# """)
    
#     results = engine.run("""
# graph TD
#     Start["{'para': {'x': 8, 'y': 4, 'z': 0}}"]
#     Multiply["{'para': {'factor': 3}}"]

#     Start -- "{'x':'a','y':'b'}" --> Compare
#     Start --> Multiply
#     Start -- "{'x':'numerator','y':'denominator'}" --> Divide
#     Multiply --> ValidateResult
#     Divide --> ValidateResult
#     ValidateResult --> End
# """)
    
#     results = engine.run("""
# graph TD
#     Start["{'para': {'x': 7, 'y': 3, 'z': 2}}"]
#     Multiply["{'para': {'factor': 3}}"]

#     Start -- "{'x':'a','y':'b'}" --> Add
#     Add -- "{'sum':'x'}" --> Multiply
#     Start -- "{'x':'a','y':'b'}" --> Subtract
#     Multiply -- "{'product':'a'}" --> Modulus
#     Subtract -- "{'result':'b'}" --> Modulus
#     Modulus -- "{'remainder':'product'}" --> ValidateResult
#     ValidateResult --> End
# """)

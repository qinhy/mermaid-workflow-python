import re
from typing import Dict, Any
from pydantic import BaseModel

# -------- Mermaid definition --------
mermaid_definition = """
graph TD
    A["()->LoadXOutput"] --> B
    D["()->LoadYOutput"] --> B
    B["(AddInputs)->AddResult"] --> C
    C["(AddResult)->"]
"""

# -------- Pydantic Models with Descriptive Names --------
class LoadXOutput(BaseModel):
    x: int

class LoadYOutput(BaseModel):
    y: int

class AddInputs(BaseModel):
    x: int
    y: int

class AddResult(BaseModel):
    z: int

# -------- Functions --------
def A() -> LoadXOutput:
    print("A: loading x")
    return LoadXOutput(x=10)

def D() -> LoadYOutput:
    print("D: loading y")
    return LoadYOutput(y=32)

def B(x: int, y: int) -> AddResult:
    print(f"B: adding {x} + {y}")
    return AddResult(z=x + y)

def C(z: int):
    print(f"C: final result is {z}")
    return None

# -------- Registry --------
node_functions = {
    'A': A,
    'B': B,
    'C': C,
    'D': D
}

model_registry = {
    'AddInputs': AddInputs,
    'AddResult': AddResult,
    'LoadXOutput': LoadXOutput,
    'LoadYOutput': LoadYOutput,
}

# -------- Parser --------
def parse_mermaid_with_models(mermaid_str):
    node_pattern = re.findall(r'(\w+)\s*\["\((.*?)\)->(.*?)"\]', mermaid_str)
    edge_pattern = re.findall(r'(\w+)\s*-->\s*(\w+)', mermaid_str)

    graph = {}
    metadata = {}

    for src, dst in edge_pattern:
        graph.setdefault(src, []).append(dst)

    for node_id, input_model, output_model in node_pattern:
        metadata[node_id] = {
            'input_model': input_model.strip() or None,
            'output_model': output_model.strip() or None,
        }

    return graph, metadata

# -------- Runner --------
def run_workflow(graph, metadata, start_nodes):
    visited = set()
    queue = list(start_nodes)
    global_context = {}

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue

        func = node_functions.get(node)
        meta = metadata[node]
        input_model = model_registry.get(meta['input_model']) if meta['input_model'] else None
        output_model = model_registry.get(meta['output_model']) if meta['output_model'] else None

        # Prepare inputs
        if input_model:
            required_fields = list(input_model.model_fields.keys())
            if not all(k in global_context for k in required_fields):
                print(f"⏳ Waiting for {node}: missing {[k for k in required_fields if k not in global_context]}")
                queue.append(node)
                continue
            input_data = {k: global_context[k] for k in required_fields}
            validated_input = input_model(**input_data)
        else:
            validated_input = None

        visited.add(node)
        print(f"\n▶ Running {node} ({meta['input_model']})->{meta['output_model']}")
        if validated_input:
            result = func(**validated_input.model_dump())
        else:
            result = func()

        if result and output_model:
            if not isinstance(result, output_model):
                raise TypeError(f"❌ {node} returned wrong type: expected {output_model}, got {type(result)}")
            global_context.update(result.model_dump())

        for neighbor in graph.get(node, []):
            if neighbor not in queue:
                queue.append(neighbor)

# -------- Main --------
if __name__ == "__main__":
    graph, metadata = parse_mermaid_with_models(mermaid_definition)
    all_targets = {dst for dsts in graph.values() for dst in dsts}
    start_nodes = [node for node in metadata if node not in all_targets]
    run_workflow(graph, metadata, start_nodes)

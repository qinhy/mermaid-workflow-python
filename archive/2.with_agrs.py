import re
import inspect

# -------- Mermaid workflow definition --------
mermaid_definition = """
graph TD
    A[Load X] --> C[Add]
    B[Load Y] --> C
    C --> D[Show Result]
"""

# -------- Functions with multiple inputs --------
def A():
    print("Loading X")
    return {"x": 10}

def B():
    print("Loading Y")
    return {"y": 32}

def C(x, y):
    print(f"Adding: {x} + {y}")
    return {"z": x + y}

def D(z):
    print("Result is:", z)
    return {}

# -------- Node to function map --------
node_functions = {
    'A': A,
    'B': B,
    'C': C,
    'D': D
}

# -------- Mermaid parser --------
def parse_mermaid(mermaid_str):
    cleaned = re.sub(r'(\w+)\[.*?\]', r'\1', mermaid_str)
    edges = re.findall(r'(\w+)\s*-->\s*(\w+)', cleaned)
    graph = {}
    for src, dst in edges:
        graph.setdefault(src, []).append(dst)
    return graph

# -------- Workflow runner with multi-arg support --------
def run_workflow(graph, start_nodes):
    visited = set()
    queue = list(start_nodes)
    context_map = {}  # outputs per node
    all_outputs = {}  # global key-value store

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        func = node_functions.get(node)
        if not func:
            print(f"Warning: No function for node {node}")
            continue

        # Inspect function args and build kwargs
        sig = inspect.signature(func)
        required_args = sig.parameters.keys()
        kwargs = {arg: all_outputs[arg] for arg in required_args if arg in all_outputs}

        print(f"\nRunning {node} with args: {kwargs}")
        result = func(**kwargs)

        if result:
            all_outputs.update(result)
            context_map[node] = result

        for neighbor in graph.get(node, []):
            queue.append(neighbor)

# -------- Entry point --------
if __name__ == "__main__":
    graph = parse_mermaid(mermaid_definition)
    
    # Automatically find root nodes (no incoming edges)
    all_targets = {dst for srcs in graph.values() for dst in srcs}
    start_nodes = [node for node in graph if node not in all_targets]
    
    run_workflow(graph, start_nodes)

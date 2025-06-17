import re
import inspect

# -------- Mermaid workflow definition with typed signatures --------
mermaid_definition = """
graph TD
    A["()->{x:int}"] --> B
    B["(x:int, y:int)->{z:int}"] --> C
    C["(z:int)->{}"]
    D["()->{y:int}"] --> B
"""

# -------- Sample functions matching the declared signatures --------
def A():
    print("Running A")
    return {"x": 10}

def D():
    print("Running D")
    return {"y": 32}

def B(x, y):
    print(f"Running B: {x} + {y}")
    return {"z": x + y}

def C(z):
    print("Final result:", z)
    return {}

# -------- Node-function map --------
node_functions = {
    'A': A,
    'B': B,
    'C': C,
    'D': D
}

# -------- Signature parser --------
def parse_signature(sig_str):
    """
    Parses a string like (x:int, y:int)->{z:int}
    Returns: (input_names, output_names)
    """
    input_match = re.search(r'\((.*?)\)', sig_str)
    output_match = re.search(r'\{(.*?)\}', sig_str)

    def parse_params(param_str):
        if not param_str or param_str.strip() == '':
            return []
        return [p.strip().split(':')[0] for p in param_str.split(',') if ':' in p]

    inputs = parse_params(input_match.group(1) if input_match else '')
    outputs = parse_params(output_match.group(1) if output_match else '')
    return inputs, outputs

# -------- Mermaid parser with typed nodes --------
def parse_mermaid_with_signatures(mermaid_str):
    node_pattern = re.findall(r'(\w+)\s*\["(.*?)"\]', mermaid_str)
    edge_pattern = re.findall(r'(\w+)\s*-->\s*(\w+)', mermaid_str)

    graph = {}
    metadata = {}

    for src, dst in edge_pattern:
        graph.setdefault(src, []).append(dst)

    for node_id, sig_label in node_pattern:
        inputs, outputs = parse_signature(sig_label)
        metadata[node_id] = {
            "inputs": inputs,
            "outputs": outputs,
            "signature": sig_label
        }

    return graph, metadata

# -------- Workflow runner --------
def run_workflow(graph, metadata, start_nodes):
    visited = set()
    queue = list(start_nodes)
    all_outputs = {}  # shared context for function args

    while queue:
        node = queue.pop(0)

        if node in visited:
            continue

        func = node_functions.get(node)
        meta = metadata[node]
        required_args = meta["inputs"]

        # Check if all required inputs are available
        if not all(arg in all_outputs for arg in required_args):
            print(f"⏳ Waiting for inputs to run {node}: missing {[arg for arg in required_args if arg not in all_outputs]}")
            queue.append(node)  # re-queue for later
            continue

        visited.add(node)
        kwargs = {arg: all_outputs[arg] for arg in required_args}

        print(f"\n▶ Running {node} with {meta['signature']}")
        print(f"  → Args: {kwargs}")
        result = func(**kwargs)

        if result:
            missing = [key for key in meta["outputs"] if key not in result]
            if missing:
                print(f"⚠️ Node {node} missing outputs: {missing}")
            all_outputs.update(result)

        for neighbor in graph.get(node, []):
            if neighbor not in queue:
                queue.append(neighbor)

# -------- Run it --------
if __name__ == "__main__":
    graph, metadata = parse_mermaid_with_signatures(mermaid_definition)

    # Detect start nodes (no incoming edges)
    all_targets = {dst for dsts in graph.values() for dst in dsts}
    start_nodes = [node for node in metadata if node not in all_targets]

    run_workflow(graph, metadata, start_nodes)

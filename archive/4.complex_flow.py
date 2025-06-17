import re
from typing import List
from pydantic import BaseModel

# -------- Mermaid definition --------
mermaid_definition = """
graph TD
    load_user_data["()->LoadUserData"] --> validate_user
    load_transactions["()->LoadTransactionData"] --> validate_transactions
    validate_user["(UserData)->ValidatedUserData"] --> merge_user_and_tx
    validate_transactions["(TransactionData)->ValidatedTransactionData"] --> merge_user_and_tx
    merge_user_and_tx["(ValidatedUserData, ValidatedTransactionData)->MergedData"] --> transform_summary
    transform_summary["(MergedData)->TransformedData"] --> encode_summary
    encode_summary["(TransformedData)->EncodedData"] --> store_data
    store_data["(EncodedData)->"] --> report_data
    store_data --> audit_data
    report_data["(EncodedData)->"]
    audit_data["(EncodedData)->"]
"""

# -------- Pydantic Models --------
class LoadUserData(BaseModel):
    user_id: int
    name: str

class LoadTransactionData(BaseModel):
    transactions: List[float]

class UserData(BaseModel):
    user_id: int
    name: str

class TransactionData(BaseModel):
    transactions: List[float]

class ValidatedUserData(BaseModel):
    user_id: int
    name: str

class ValidatedTransactionData(BaseModel):
    transactions: List[float]

class MergedData(BaseModel):
    user_id: int
    name: str
    total: float

class TransformedData(BaseModel):
    summary: str

class EncodedData(BaseModel):
    blob: str

# -------- Node functions --------
def load_user_data() -> LoadUserData:
    print("A: Loading user data")
    return LoadUserData(user_id=1, name="Alice")

def load_transactions() -> LoadTransactionData:
    print("B: Loading transaction data")
    return LoadTransactionData(transactions=[100.0, 200.0, 50.0])

def validate_user(user_id: int, name: str) -> ValidatedUserData:
    print("C: Validating user")
    return ValidatedUserData(user_id=user_id, name=name.title())

def validate_transactions(transactions: List[float]) -> ValidatedTransactionData:
    print("D: Validating transactions")
    return ValidatedTransactionData(transactions=[t for t in transactions if t > 0])

def merge_user_and_tx(user_id: int, name: str, transactions: List[float]) -> MergedData:
    total = sum(transactions)
    print(f"E: Merging user {name} with total ${total}")
    return MergedData(user_id=user_id, name=name, total=total)

def transform_summary(user_id: int, name: str, total: float) -> TransformedData:
    summary = f"{name} (ID: {user_id}) spent ${total:.2f}"
    print("F: Transforming data")
    return TransformedData(summary=summary)

def encode_summary(summary: str) -> EncodedData:
    print("G: Encoding summary")
    return EncodedData(blob=summary.encode("utf-8").hex())

def store_data(blob: str):
    print("H: Storing encoded data:", blob)

def report_data(blob: str):
    print("I: Reporting:", blob)

def audit_data(blob: str):
    print("J: Auditing:", blob)

# -------- Function and model registry --------
node_functions = {
    "load_user_data":load_user_data,
    "load_transactions":load_transactions,
    "validate_user":validate_user,
    "validate_transactions":validate_transactions,
    "merge_user_and_tx":merge_user_and_tx,
    "transform_summary":transform_summary,
    "encode_summary":encode_summary,
    "store_data":store_data,
    "report_data":report_data,
    "audit_data":audit_data,
}

model_registry = {
    'LoadUserData': LoadUserData,
    'LoadTransactionData': LoadTransactionData,
    'UserData': UserData,
    'TransactionData': TransactionData,
    'ValidatedUserData': ValidatedUserData,
    'ValidatedTransactionData': ValidatedTransactionData,
    'MergedData': MergedData,
    'TransformedData': TransformedData,
    'EncodedData': EncodedData
}

# -------- Mermaid parser --------
def parse_mermaid_with_models(mermaid_str):
    node_pattern = re.findall(r'(\w+)\s*\["\((.*?)\)->(.*?)"\]', mermaid_str)
    edge_pattern = re.findall(r'(\w+)\s*-->\s*(\w+)', mermaid_str)

    graph = {}
    metadata = {}

    for src, dst in edge_pattern:
        graph.setdefault(src, []).append(dst)

    for node_id, input_model_str, output_model in node_pattern:
        input_models = [m.strip() for m in input_model_str.split(',') if m.strip()]
        metadata[node_id] = {
            'input_models': input_models,
            'output_model': output_model.strip() or None,
        }

    return graph, metadata

# -------- Workflow runner --------
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
        input_model_names = meta['input_models']
        output_model_name = meta['output_model']

        input_fields = []
        input_models = []

        # Gather all input fields from all models
        for model_name in input_model_names:
            if model_name not in model_registry:
                raise ValueError(f"❌ Unknown model '{model_name}' for node {node}")
            model = model_registry[model_name]
            input_models.append(model)
            input_fields.extend(model.model_fields.keys())

        missing_fields = [f for f in input_fields if f not in global_context]
        if missing_fields:
            print(f"⏳ Waiting for {node}: missing {missing_fields}")
            queue.append(node)
            continue

        # Build combined input data
        combined_inputs = {k: global_context[k] for k in input_fields}

        visited.add(node)
        print(f"\n▶ Running {node} ({', '.join(input_model_names)}) -> {output_model_name}")
        result = func(**combined_inputs)

        # Process outputs
        if output_model_name:
            output_model = model_registry.get(output_model_name)
            if not isinstance(result, output_model):
                raise TypeError(f"❌ {node} returned wrong type: expected {output_model_name}, got {type(result)}")
            global_context.update(result.model_dump())
        elif isinstance(result, BaseModel):
            global_context.update(result.model_dump())

        for neighbor in graph.get(node, []):
            if neighbor not in queue:
                queue.append(neighbor)

from graphlib import TopologicalSorter, CycleError

def run_workflow(graph, metadata, start_nodes):
    ts = TopologicalSorter(graph)
    try:
        ts.prepare()
    except CycleError as e:
        raise ValueError(f"❌ Cycle detected in workflow: {e}")

    global_context = {}

    while ts.is_active():
        ready_nodes = list(ts.get_ready())

        for node in ready_nodes:
            meta = metadata[node]
            input_model_names = meta['input_models']
            output_model_name = meta['output_model']
            func = node_functions[node]

            input_fields = []
            for model_name in input_model_names:
                if model_name not in model_registry:
                    raise ValueError(f"❌ Unknown model '{model_name}' for node {node}")
                model = model_registry[model_name]
                input_fields.extend(model.model_fields.keys())

            if any(f not in global_context for f in input_fields):
                ts.defer(node)
                continue

            print(f"\n▶ Running {node} ({', '.join(input_model_names)}) -> {output_model_name}")
            combined_inputs = {k: global_context[k] for k in input_fields}
            result = func(**combined_inputs)

            if output_model_name:
                output_model = model_registry[output_model_name]
                if not isinstance(result, output_model):
                    raise TypeError(f"❌ {node} returned wrong type: expected {output_model_name}, got {type(result)}")
                global_context.update(result.model_dump())
            elif isinstance(result, BaseModel):
                global_context.update(result.model_dump())

            ts.done(node)
# -------- Main --------
if __name__ == "__main__":
    graph, metadata = parse_mermaid_with_models(mermaid_definition)
    all_targets = {dst for dsts in graph.values() for dst in dsts}
    start_nodes = [node for node in metadata if node not in all_targets]
    run_workflow(graph, metadata, start_nodes)

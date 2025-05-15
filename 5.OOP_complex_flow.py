
import json
from typing import List, Dict, Any, Type
from pydantic import BaseModel
import re
from collections import defaultdict
from graphlib import TopologicalSorter

# -------- Mermaid definition --------
mermaid_definition = """
graph TD
    LoadUserData["{'pars':{'endpoint':'default_user_endpoint'}}"]
    LoadUserData --> ValidateUser
    LoadTransactions --> ValidateTransactions

    ValidateUser --> MergeUserAndTx
    ValidateTransactions --> MergeUserAndTx

    MergeUserAndTx --> TransformSummary
    TransformSummary --> EncodeSummary
    EncodeSummary --> StoreData

    MergeUserAndTx --> StoreData
    StoreData --> ReportData
    StoreData --> AuditData
"""

def parse_mermaid_with_models(mermaid_text: str) -> Dict[str, Dict[str, Any]]:
    lines = [line.strip() for line in mermaid_text.strip().splitlines()]
    
    graph = defaultdict(lambda: {"prev": [], "next": [], "config": None})
    
    # Pattern to detect node with config: NodeName["<json_string>"]
    config_pattern = re.compile(r'^(\w+)\s*\[\s*"(.+)"\s*\]$')
    edge_pattern = re.compile(r"(\w+)\s*-->\s*(\w+)")

    # First pass: extract node configs
    for line in lines:
        if '-->' not in line:
            config_match = config_pattern.match(line)
            if config_match:
                node, config_str = config_match.groups()
                try:
                    graph[node]["config"] = json.loads(config_str.replace("'", '"'))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON for node '{node}': {e}")

    # Second pass: extract edges
    for line in lines:
        if '-->' not in line:
            continue

        match = edge_pattern.match(line)
        if match:
            src, dst = match.groups()
            graph[src]["next"].append(dst)
            graph[dst]["prev"].append(src)
        else:
            raise ValueError(f"Invalid Mermaid edge syntax: {line}")

    return dict(graph)

# -------- Pydantic Models --------
class LoadUserData:
    class Pars(BaseModel):
        endpoint: str = "NULL"

    class Args(BaseModel):
        pass

    class Rets(BaseModel):
        user_id: int
        name: str

    pars:Pars|None = None
    args:Args|None = None
    rets:Rets|None = None

    def __call__(self) -> 'LoadUserData.Rets':
        print("A: Loading user data")
        self.rets = self.Rets(user_id=1, name="Alice")
        
class LoadTransactions:
    class Pars(BaseModel):
        source: str = "default_transaction_source"

    class Args(BaseModel):
        pass

    class Rets(BaseModel):
        transactions: List[float]

    pars:Pars|None = None
    args:Args|None = None
    rets:Rets|None = None

    def __call__(self) -> 'LoadTransactions.Rets':
        print("B: Loading transaction data")
        self.rets = self.Rets(transactions=[100.0, 200.0, 50.0])

class ValidateUser:
    class Args(BaseModel):
        user_id: int
        name: str

    class Rets(BaseModel):
        user_id: int
        name: str

    args:Args|None = None
    rets:Rets|None = None

    def __call__(self):
        print("C: Validating user")
        self.rets = self.Rets(user_id=self.args.user_id, name=self.args.name.title())
        
class ValidateTransactions:
    class Args(BaseModel):
        transactions: List[float]

    class Rets(BaseModel):
        transactions: List[float]

    args:Args|None = None
    rets:Rets|None = None

    def __call__(self):
        print("D: Validating transactions")
        self.rets = self.Rets(transactions=[t for t in self.args.transactions if t > 0])
        
class MergeUserAndTx:
    class Args(BaseModel):
        user_id: int
        name: str
        transactions: List[float]

    class Rets(BaseModel):
        user_id: int
        name: str
        total: float

    args:Args|None = None
    rets:Rets|None = None

    def __call__(self):
        total = sum(self.args.transactions)
        print(f"E: Merging user {self.args.name} with total ${total}")
        self.rets = self.Rets(user_id=self.args.user_id, name=self.args.name, total=total)
        
class TransformSummary:
    class Args(BaseModel):
        user_id: int
        name: str
        total: float

    class Rets(BaseModel):
        summary: str

    args:Args|None = None
    rets:Rets|None = None

    def __call__(self):
        print("F: Transforming data")
        summary = f"{self.args.name} (ID: {self.args.user_id}) spent ${self.args.total:.2f}"
        self.rets = self.Rets(summary=summary)
        
class EncodeSummary:
    class Args(BaseModel):
        summary: str

    class Rets(BaseModel):
        blob: str

    args:Args|None = None
    rets:Rets|None = None

    def __call__(self):
        print("G: Encoding summary")
        self.rets = self.Rets(blob=self.args.summary.encode("utf-8").hex())
        
class StoreData:
    class Args(BaseModel):
        blob: str

    class Rets(BaseModel):
        blob: str

    args: Args | None = None
    rets: Rets | None = None

    def __call__(self):
        print("H: Storing encoded data:", self.args.blob)
        self.rets = self.Rets(blob=self.args.blob)

        
class ReportData:
    class Args(BaseModel):
        blob: str

    args:Args|None = None
    
    def __call__(self):
        print("I: Reporting:", self.args.blob)
        
class AuditData:
    class Args(BaseModel):
        blob: str

    args:Args|None = None
    
    def __call__(self):
        print("J: Auditing:", self.args.blob)

# -------- Function and model registry --------

model_registry = {
    LoadUserData:"LoadUserData",
    LoadTransactions:"LoadTransactions",
    ValidateUser:"ValidateUser",
    ValidateTransactions:"ValidateTransactions",
    MergeUserAndTx:"MergeUserAndTx",
    TransformSummary:"TransformSummary",
    EncodeSummary:"EncodeSummary",
    StoreData:"StoreData",
    ReportData:"ReportData",
    AuditData:"AuditData",
}

def collect_args_from_deps(
    node_name: str,
    args_fields: List[str],
    deps: List[str],
    results: Dict[str, Any]
) -> Dict[str, Any]:
    args_data = {}
    used_fields = set()
    field_source_map = defaultdict(list)  # Tracks which fields came from which dep

    # Collect fields from all deps
    for dep in deps:
        dep_result = results.get(dep, {})
        for key in dep_result:
            field_source_map[dep].append(key)

    # Try to match expected args
    for field in args_fields:
        found = False
        for dep in deps:
            dep_result = results.get(dep, {})
            if field in dep_result:
                args_data[field] = dep_result[field]
                used_fields.add(field)
                found = True
                break
        if not found:
            print(f"⚠️ Warning: Field '{field}' required by '{node_name}' not found in dependencies: {deps}")

    # Identify unused fields per dependency
    for dep, fields in field_source_map.items():
        unused = set(fields) - used_fields
        if unused:
            print(f"ℹ️ Info: Fields from '{dep}' not used by '{node_name}': {sorted(unused)}")

    return args_data

# -------- Workflow runner --------
def run_workflow(mermaid_text: str, model_registry: Dict[Type, str]):
    # --- Step 1: Parse the Mermaid text into a graph structure ---
    graph = parse_mermaid_with_models(mermaid_text)

    # --- Step 2: Create a reverse lookup: name -> class ---
    name_to_class = {v: k for k, v in model_registry.items()}

    # --- Step 3: Build a topological sorter with dependencies ---
    ts = TopologicalSorter()
    for node, deps in graph.items():
        ts.add(node, *deps["prev"])
    ts.prepare()

    # --- Step 4: Container for storing results of each step ---
    results: Dict[str, Any] = {}

    # --- Step 5: Execute steps in topological order ---
    while ts.is_active():
        ready_nodes = ts.get_ready()

        for node_name in ready_nodes:
            cls = name_to_class[node_name]
            instance = cls()

            # --- Step 5a: Gather input arguments from dependencies ---
            deps = graph[node_name]["prev"]
            conf = graph[node_name]["config"]
            
            if hasattr(cls, 'Pars'):
                if conf and 'pars' in conf:
                    instance.pars = cls.Pars(**conf['pars'])

            if hasattr(cls, 'Args'):
                args_fields = cls.Args.model_fields.keys()
                args_data = collect_args_from_deps(
                    node_name=node_name,
                    args_fields=list(args_fields),
                    deps=deps,
                    results=results
                )

                if conf and 'args' in conf:
                    args_data.update(conf['args'])

                instance.args = cls.Args(**args_data)

            # --- Step 5b: Run the step logic ---
            output = instance()

            # --- Step 5c: Store the result from the step ---
            if hasattr(instance, 'rets') and isinstance(instance.rets, BaseModel):
                results[node_name] = instance.rets.model_dump()
            else:
                results[node_name] = {}  # explicitly no output

            # --- Mark the step as completed ---
            ts.done(node_name)

    # --- Step 6: Final output summary ---
    print("\n✅ Final outputs:")
    for step_name, output in results.items():
        print(f"{step_name}: {output}")

    return results

def validate_workflow_io(mermaid_text: str, model_registry: Dict[Type, str]) -> bool:
    print("\n🔍 Validating workflow I/O...")
    graph = parse_mermaid_with_models(mermaid_text)
    name_to_class = {v: k for k, v in model_registry.items()}
    all_classes = [i for i in graph.keys() if i not in name_to_class]
    if len(all_classes) > 0:
        print(f"❌ Unknown classes found: {all_classes}")
        return False

    all_valid = True

    for node_name, meta in graph.items():
        deps = meta["prev"]
        cls = name_to_class.get(node_name)

        if not hasattr(cls, "Args"):
            continue  # Node takes no input
        required_fields = cls.Args.model_fields.keys()
        field_to_sources = defaultdict(list)
        field_source_map = defaultdict(list)

        # Collect all available output fields from deps
        for dep in deps:
            dep_cls = name_to_class.get(dep)
            if dep_cls and hasattr(dep_cls, "Rets"):
                for field in dep_cls.Rets.model_fields.keys():
                    field_to_sources[field].append(dep)
                    field_source_map[dep].append(field)

        used_fields = set()

        # Check for missing fields and mark used ones
        for field in required_fields:
            if field in field_to_sources:
                used_fields.add(field)
            else:
                print(f"❌ Missing field '{field}' for node '{node_name}' (from deps: {deps})")
                all_valid = False

        # Warn about conflicting fields (same field from multiple deps)
        for field, sources in field_to_sources.items():
            if len(sources) > 1 and field in required_fields:
                print(f"⚠️ Warning: Field '{field}' for node '{node_name}' comes from multiple sources: {sources}")

        # Info about unused fields from each dep
        for dep, fields in field_source_map.items():
            unused = set(fields) - used_fields
            if unused:
                print(f"ℹ️ Info: Fields from '{dep}' not used by '{node_name}': {sorted(unused)}")

    if all_valid:
        print("\n✅ Workflow validation passed: All inputs have matching outputs.")
    else:
        print("\n❌ Workflow validation failed. Review issues above.")

    return all_valid

# -------- Main --------
if __name__ == "__main__":
    valid = validate_workflow_io(mermaid_definition,model_registry)
    run_workflow(mermaid_definition,model_registry)

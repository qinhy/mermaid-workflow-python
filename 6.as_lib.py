
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

class MermaidWorkflowEngine:
    def __init__(self, mermaid_text: str, model_registry: Dict[Type, str]):
        self.mermaid_text = mermaid_text
        self.model_registry = model_registry
        self.name_to_class = {v: k for k, v in model_registry.items()}
        self.graph = self._parse_mermaid()
        self.results: Dict[str, Any] = {}

    def _parse_mermaid(self) -> Dict[str, Dict[str, Any]]:
        lines = [line.strip() for line in self.mermaid_text.strip().splitlines()]
        graph = defaultdict(lambda: {"prev": [], "next": [], "config": None})

        config_pattern = re.compile(r'^(\w+)\s*\[\s*"(.+)"\s*\]$')
        edge_pattern = re.compile(r"(\w+)\s*-->\s*(\w+)")

        for line in lines:
            if '-->' not in line:
                config_match = config_pattern.match(line)
                if config_match:
                    node, config_str = config_match.groups()
                    try:
                        graph[node]["config"] = json.loads(config_str.replace("'", '"'))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON for node '{node}': {e}")

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

    def _collect_args_from_deps(self, node_name: str, args_fields: List[str], deps: List[str]) -> Dict[str, Any]:
        args_data = {}
        used_fields = set()
        field_source_map = defaultdict(list)

        for dep in deps:
            dep_result = self.results.get(dep, {})
            for key in dep_result:
                field_source_map[dep].append(key)

        for field in args_fields:
            found = False
            for dep in deps:
                dep_result = self.results.get(dep, {})
                if field in dep_result:
                    args_data[field] = dep_result[field]
                    used_fields.add(field)
                    found = True
                    break
            if not found:
                print(f"⚠️ Warning: Field '{field}' required by '{node_name}' not found in dependencies: {deps}")

        for dep, fields in field_source_map.items():
            unused = set(fields) - used_fields
            if unused:
                print(f"ℹ️ Info: Fields from '{dep}' not used by '{node_name}': {sorted(unused)}")

        return args_data
    
    def __node_fields(self, node_name: str, field_name: str):
        cls = self.name_to_class.get(node_name)
        if not hasattr(cls, field_name): return []
        cls = cls.__dict__[field_name]
        if not hasattr(cls, "model_fields"): return []
        return cls.model_fields.keys()

    def __node_args_fields(self, node_name: str):
        return self.__node_fields(node_name, "Args")

    def __node_rets_fields(self, node_name: str):
        return self.__node_fields(node_name, "Rets")

    def validate_io(self) -> bool:
        print("\n🔍 Validating workflow I/O...")
        all_classes = [i for i in self.graph.keys() if i not in self.name_to_class]
        if all_classes:
            print(f"❌ Unknown classes found: {all_classes}")
            return False

        all_valid = True

        for node_name, meta in self.graph.items():
            deps = meta["prev"]
            required_fields = self.__node_args_fields(node_name)
            field_to_sources = defaultdict(list)
            field_source_map = defaultdict(list)

            for dep in deps:
                for field in self.__node_rets_fields(dep):
                    field_to_sources[field].append(dep)
                    field_source_map[dep].append(field)

            used_fields = set()

            for field in required_fields:
                if field in field_to_sources:
                    used_fields.add(field)
                else:
                    print(f"❌ Missing field '{field}' for node '{node_name}' (from deps: {deps})")
                    all_valid = False

            for field, sources in field_to_sources.items():
                if len(sources) > 1 and field in required_fields:
                    print(f"⚠️ Warning: Field '{field}' for node '{node_name}' comes from multiple sources: {sources}")

            for dep, fields in field_source_map.items():
                unused = set(fields) - used_fields
                if unused:
                    print(f"ℹ️ Info: Fields from '{dep}' not used by '{node_name}': {sorted(unused)}")

        if all_valid:
            print("\n✅ Workflow validation passed: All inputs have matching outputs.")
        else:
            print("\n❌ Workflow validation failed. Review issues above.")
        return all_valid

    def run(self) -> Dict[str, Any]:
        ts = TopologicalSorter()
        for node, meta in self.graph.items():
            ts.add(node, *meta["prev"])
        ts.prepare()

        while ts.is_active():
            for node_name in ts.get_ready():
                cls = self.name_to_class[node_name]
                instance = cls()

                deps = self.graph[node_name]["prev"]
                conf = self.graph[node_name]["config"]

                if hasattr(cls, "Pars") and conf and 'pars' in conf:
                    instance.pars = cls.Pars(**conf['pars'])

                args_fields = self.__node_args_fields(node_name)
                if args_fields:
                    args_data = self._collect_args_from_deps(node_name, list(args_fields), deps)

                    if conf and 'args' in conf:
                        args_data.update(conf['args'])

                    instance.args = cls.Args(**args_data)

                output = instance()

                if hasattr(instance, "rets") and isinstance(instance.rets, BaseModel):
                    self.results[node_name] = instance.rets.model_dump()
                else:
                    self.results[node_name] = {}

                ts.done(node_name)

        print("\n✅ Final outputs:")
        for step_name, output in self.results.items():
            print(f"{step_name}: {output}")

        return self.results


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

# -------- Main --------
if __name__ == "__main__":
    engine = MermaidWorkflowEngine(mermaid_definition, model_registry)

    # Optional validation
    if engine.validate_io():
        results = engine.run()

import json
from typing import List, Dict, Any, Type
from pydantic import BaseModel
import re
from collections import defaultdict
from graphlib import TopologicalSorter


class MermaidWorkflowFunction(BaseModel):
    """Base class for workflow function nodes with parameters, arguments and returns."""
    
    class Parameters(BaseModel):
        """Static parameters that configure the function behavior."""
        pass

    class Arguments(BaseModel):
        """Input arguments received from predecessor nodes."""
        pass

    class Returns(BaseModel):
        """Output values passed to successor nodes."""
        pass

    para: Parameters | None = None
    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self) -> Returns:
        """Execute the workflow function and return results.
        
        Returns:
            Returns: Output values to be passed to successor nodes
        """
        raise NotImplementedError("Workflow functions must implement __call__")

    def update_para(self, data: dict) -> None:
        # data must has key of "para"
        if not data or "para" not in data : return
        para = self.para.model_dump()
        para.update(data["para"])
        self.para = self.Parameters(**para)

    def update_args(self, data: dict) -> None:
        # data must has key of "args"
        if not data or "args" not in data : return
        args = self.args.model_dump()
        args.update(data["args"])
        self.args = self.Arguments(**args)

    def update_rets(self, data: dict) -> None:
        # data must has key of "rets"
        if not data or "rets" not in data : return
        rets = self.rets.model_dump()
        rets.update(data["rets"])
        self.rets = self.Returns(**rets)

    @classmethod
    def get_para_class_name(cls) -> str:
        """Get the name of the Parameters class."""
        return cls.Parameters.__name__

    @classmethod
    def get_args_class_name(cls) -> str:
        """Get the name of the Arguments class."""
        return cls.Arguments.__name__

    @classmethod
    def get_rets_class_name(cls) -> str:
        """Get the name of the Returns class."""
        return cls.Returns.__name__ 

    @classmethod
    def get_model_fields(cls, model_name: str) -> list[str]:
        """Get field names from a specified model class.
        
        Args:
            model_name: Name of the model class ('Parameters', 'Arguments', or 'Returns')
            
        Returns:
            List of field names defined in the model
        """
        if not hasattr(cls, model_name):
            return []
        model_cls:BaseModel = getattr(cls, model_name)
        if not hasattr(model_cls, "model_fields"):
            return []
        return list(model_cls.model_fields.keys())

    @classmethod
    def get_argument_fields(cls) -> list[str]:
        """Get names of all input argument fields."""
        return cls.get_model_fields(cls.get_args_class_name())

    @classmethod
    def get_return_fields(cls) -> list[str]:
        """Get names of all return value fields."""
        return cls.get_model_fields(cls.get_rets_class_name())
    
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
    
    def __node_args_fields(self, node_name: str):
        cls:MermaidWorkflowFunction = self.name_to_class.get(node_name)
        return cls.get_argument_fields()

    def __node_rets_fields(self, node_name: str):
        cls:MermaidWorkflowFunction = self.name_to_class.get(node_name)
        return cls.get_return_fields()

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
                self.results[node_name] = {}

                cls:MermaidWorkflowFunction = self.name_to_class[node_name]
                instance:MermaidWorkflowFunction = cls()

                deps = self.graph[node_name]["prev"]
                args_fields = self.__node_args_fields(node_name)
                if args_fields:
                    args_data = self._collect_args_from_deps(node_name, list(args_fields), deps)
                    instance.args = cls.Arguments(**args_data)
                
                conf:dict = self.graph[node_name]["config"]
                instance.update_para(conf)
                instance.update_args(conf)
                instance()
                if hasattr(instance, "rets") and hasattr(instance.rets, 'model_dump'):
                    self.results[node_name] = instance.rets.model_dump()

                ts.done(node_name)

        print("\n✅ Final outputs:")
        for step_name, output in self.results.items():
            print(f"{step_name}: {output}")

        return self.results
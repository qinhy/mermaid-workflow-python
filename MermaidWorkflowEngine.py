from collections import defaultdict
from graphlib import TopologicalSorter
from pydantic import BaseModel, create_model
from typing import (
    Any, 
    Annotated,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union
)
import json
import re


def parse_mermaid(mermaid_text: str) -> dict:
    lines = [l.strip() for l in mermaid_text.strip().splitlines()]
    lines = [l for l in lines if ('["{' in l) or ('--' in l)]
    lines = [l for l in lines if '["{}"]' not in l]
    lines = [l for l in lines if not l.startswith('%%')]
    graph = defaultdict(lambda: {"prev": [], "next": [], "config": {}, "maps": {}})

    node_pattern = re.compile(r'^([\w-]+)\s*\[\s*"(.+)"\s*\]$')
    map_pattern = re.compile(r'^([\w-]+)\s*--\s*"(.*?)"\s*-->\s*([\w-]+)$')
    simple_pattern = re.compile(r'^([\w-]+)\s*-->\s*([\w-]+)$')

    def parse_json(s: str) -> Any:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return None

    for l in lines:
        if not l or l.startswith("graph"):
            continue
        m = node_pattern.match(l)
        if m:
            node, cfg = m.groups()
            parsed = parse_json(cfg)
            if parsed is not None:
                graph[node]["config"] = parsed
            continue
        m = map_pattern.match(l)
        if m:
            src, cfg, dst = m.groups()
            graph[src]["next"].append(dst)
            graph[dst]["prev"].append(src)
            parsed = parse_json(cfg)
            if parsed is not None:
                graph[src]["maps"][dst] = parsed
            continue
        m = simple_pattern.match(l)
        if m:
            src, dst = m.groups()
            graph[src]["next"].append(dst)
            graph[dst]["prev"].append(src)
            continue
        raise ValueError(f"Invalid Mermaid syntax: {l}")
    return dict(graph)

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

    para: Optional[Parameters] = None
    args: Optional[Arguments] = None
    rets: Optional[Returns] = None

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
    def from_mcp(cls, name: str, mcp_single_func_data: dict) -> 'MermaidWorkflowFunction':
        # Helper: build model with proper type annotations
        def build_model_from_properties(name: str, props: Dict[str, Any], required: list) -> Type[BaseModel]:
            fields: Dict[str, Tuple[Any, Any]] = {}
            for key, spec in props.items():
                # Handle anyOf case for union types
                if "anyOf" in spec:
                    types = []
                    for type_spec in spec["anyOf"]:
                        if type_spec.get("type") == "integer":
                            types.append(int)
                        elif type_spec.get("type") == "number":
                            types.append(float)
                        elif type_spec.get("type") == "string":
                            types.append(str)
                        elif type_spec.get("type") == "boolean":
                            types.append(bool)
                        elif type_spec.get("type") == "null":
                            types.append(type(None))
                    field_type = Annotated[Union[tuple(types)], ...]
                    fields[key] = (field_type, ...)
                    continue

                # Handle regular type specifications
                try:
                    type_str = spec["type"]
                except KeyError:                    
                    print(f"Processing field '{key}' with spec: {spec}")
                    continue

                match type_str:
                    case "integer":
                        field_type = Annotated[int, ...] if key in required else Annotated[int | None, None]
                    case "number":
                        field_type = Annotated[float, ...] if key in required else Annotated[float | None, None]
                    case "string":
                        field_type = Annotated[str, ...] if key in required else Annotated[str | None, None]
                    case "boolean":
                        field_type = Annotated[bool, ...] if key in required else Annotated[bool | None, None]
                    case "array":
                        item_type = spec.get("items", {}).get("type", "any")
                        if item_type == "integer":
                            field_type = Annotated[List[int], ...] if key in required else Annotated[List[int] | None, None]
                        elif item_type == "number":
                            field_type = Annotated[List[float], ...] if key in required else Annotated[List[float] | None, None]
                        elif item_type == "string":
                            field_type = Annotated[List[str], ...] if key in required else Annotated[List[str] | None, None]
                        elif item_type == "boolean":
                            field_type = Annotated[List[bool], ...] if key in required else Annotated[List[bool] | None, None]
                        else:
                            field_type = Annotated[List[Any], ...] if key in required else Annotated[List[Any] | None, None]
                    case "object":
                        field_type = Annotated[Dict[str, Any], ...] if key in required else Annotated[Dict[str, Any] | None, None]
                    case _:
                        raise NotImplementedError(f"Type '{type_str}' not supported yet.")
                fields[key] = (field_type, ...)
            return create_model(name, **fields)

        defs = mcp_single_func_data["inputSchema"]["$defs"]
        model_creation_args = {}
        if "Parameters" in defs:
            param_def = defs["Parameters"]
            Parameters = build_model_from_properties("Parameters", param_def["properties"], param_def.get("required", []))
            model_creation_args["para"] = (Parameters,...)

        if "Arguments" in defs:
            arg_def = defs["Arguments"]
            Arguments = build_model_from_properties("Arguments", arg_def["properties"], arg_def.get("required", []))
            model_creation_args["args"] = (Arguments,...)

        if "Returns" in defs:
            arg_def = defs["Returns"]
            Returns = build_model_from_properties("Returns", arg_def["properties"], arg_def.get("required", []))
            # model_creation_args["rets"] = (Returns,...)
            
        model_cls = create_model(name, **model_creation_args,__base__=MermaidWorkflowFunction)
        if "Parameters" in defs:model_cls.Parameters = Parameters
        if "Arguments" in defs:model_cls.Arguments = Arguments
        if "Returns" in defs:model_cls.Returns = Returns

        return model_cls

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
    def __init__(self, model_registry: Dict[Type, str]):
        self.mermaid_text = ""
        self.model_registry = model_registry
        self.name_to_class = [(k,v) if type(k) is str else (v,k) for k, v in model_registry.items()]
        self.name_to_class_dict = {k:v for k,v in self.name_to_class}
        for k,v in self.name_to_class:
            if type(v) is dict:
                self.name_to_class_dict[k] = MermaidWorkflowFunction.from_mcp(k,v)
        self.name_to_class = self.name_to_class_dict
        self.graph = {}
        self.results: Dict[str, Any] = {}

    def extract_mermaid_text(self, text: str) -> str:
        """Extract Mermaid flowchart text from a given text."""
        mermaid_pattern = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)
        match = mermaid_pattern.search(text)
        if match:
            return match.group(1).strip()
        return ""

    def parse_mermaid(self, mermaid_text:str) -> Dict[str, Dict[str, Any]]:
        self.mermaid_text = mermaid_text
        return parse_mermaid(self.mermaid_text)
    
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
    
    def node_args_fields(self, node_name: str):
        cls:MermaidWorkflowFunction = self.name_to_class.get(node_name)
        return cls.get_argument_fields()

    def node_rets_fields(self, node_name: str):
        cls:MermaidWorkflowFunction = self.name_to_class.get(node_name)
        return cls.get_return_fields()

    def validate_io(self) -> bool:
        print("\n🔍 Validating workflow I/O with mapping support...")

        # 1. Unknown classes check
        unknown = set(self.graph) - set(self.name_to_class)
        if unknown:
            print(f"❌ Unknown classes found: {unknown}")
            return False

        all_valid = True

        for node, meta in self.graph.items():
            deps = meta.get("prev", [])
            required = set(self.node_args_fields(node))
            mapped_fields = set()
            field_sources = defaultdict(list)

            if not deps:
                if required:
                    print(f"❌ Node '{node}' has no dependencies but requires inputs {sorted(required)}")
                    all_valid = False
                else:
                    print(f"⚠️ Node '{node}' has no dependencies and no required inputs.")
                # continue to next node
                continue

            for dep in deps:
                dep_rets = set(self.node_rets_fields(dep))
                map_cfg = self.graph[dep].get("maps", {}).get(node, {})

                # — 1) Validate explicit mappings
                bad_srcs = set(map_cfg) - dep_rets
                if bad_srcs:
                    for src in bad_srcs:
                        print(f"❌ Field '{src}' not found in '{dep}' outputs {sorted(dep_rets)} for mapping to '{node}'")
                    all_valid = False

                # Warn on mappings to unused dst-fields
                bad_dsts = set(map_cfg.values()) - required
                if bad_dsts:
                    for dst in bad_dsts:
                        print(f"⚠️ Mapping to '{dst}' ignored—it's not a required field of '{node}'")
                
                # Warn on redundant 1:1 mappings (src == dst)
                for src, dst in map_cfg.items():
                    if src == dst and src in required:
                        print(f"⚠️ Redundant explicit mapping '{src}→{dst}' for '{node}' (1:1 mapping)")

                # — 2) Apply explicit mappings
                used_by_dep = set()
                for src, dst in map_cfg.items():
                    if src in dep_rets and dst in required:
                        mapped_fields.add(dst)
                        field_sources[dst].append(dep)
                        used_by_dep.add(src)

                # — 3) Default 1:1 mappings for anything left over
                default_mapped = (dep_rets & required) - set(map_cfg.values())
                for fld in default_mapped:
                    mapped_fields.add(fld)
                    field_sources[fld].append(dep)
                    used_by_dep.add(fld)

                # — 4) Warn about any outputs of dep that go completely unused
                unused = dep_rets - used_by_dep
                if unused:
                    print(f"⚠️ From '{dep}' → '{node}', these outputs are never used: {sorted(unused)}")

            # — 5) Check for missing required fields
            missing = required - mapped_fields
            if missing:
                for fld in sorted(missing):
                    print(f"❌ Missing field '{fld}' for node '{node}' from dependencies {deps}")
                all_valid = False

            # — 6) Warn if any field comes from multiple deps
            for fld, sources in field_sources.items():
                if len(set(sources)) > 1:
                    print(f"⚠️ Field '{fld}' for node '{node}' comes from multiple sources: {sources}")

        # — 7) Final summary
        if all_valid:
            print("\n✅ Workflow validation passed: All inputs are satisfied.")
        else:
            print("\n❌ Workflow validation failed. See messages above.")

        return all_valid


    def run(self, mermaid_text: str, ignite_func: Optional[Callable] = None) -> Dict[str, Any]:
        if ignite_func is None:
            ignite_func = lambda obj, args: obj()

        self.graph = self.parse_mermaid(mermaid_text)

        if not self.validate_io():
            print("❌ Workflow validation failed. Exiting.")
            return {}

        # Build the dependency graph for topological sorting
        ts_graph = {node: meta["prev"] for node, meta in self.graph.items()}
        sorter = TopologicalSorter(ts_graph)
        execution_order = list(sorter.static_order())

        for node_name in execution_order:
            self.results[node_name] = {}
            cls:MermaidWorkflowFunction = self.name_to_class[node_name]

            # Collect inputs from dependencies
            args_data = {}
            deps = self.graph[node_name]["prev"]
            for dep in deps:
                dep_results = self.results.get(dep, {})
                map_config:dict = self.graph[dep].get("maps", {})
                map_config:dict = map_config.get(node_name, {})

                # Default direct field matching
                for field in dep_results:
                    if field in cls.get_argument_fields():
                        args_data[field] = dep_results[field]

                # Field remapping
                for src_field, dst_field in map_config.items():
                    if src_field in dep_results:
                        args_data[dst_field] = dep_results[src_field]

            # Merge static config
            conf = self.graph[node_name].get("config", {}) or {}
            para_data = conf.get("para", {})
            args_data.update(conf.get("args", {}))

            cls_data = {}
            try:
                if hasattr(cls, "Parameters") and len(cls.Parameters.model_fields) > 0:
                    cls_data['para'] = cls.Parameters(**para_data)
                if hasattr(cls, "Arguments") and len(cls.Arguments.model_fields) > 0:
                    cls_data['args'] = cls.Arguments(**args_data)
            except Exception as e:
                print(f"❌ Error validating config for '{node_name}': {e}")
                continue

            print(f"\n🔄 Executing node '{node_name}' with: {args_data}")

            try:
                instance:MermaidWorkflowFunction = cls(**cls_data)
                res = ignite_func(instance, cls_data)

                # Extract return values
                if hasattr(instance, "rets") and hasattr(instance.rets, "model_dump"):
                    self.results[node_name] = instance.rets.model_dump()
                elif isinstance(res, dict) and "rets" in res:
                    self.results[node_name] = res["rets"]

            except Exception as e:
                print(f"❌ Error executing node '{node_name}': {e}")
                continue

        print("\n✅ Final outputs:")
        for step_name, output in self.results.items():
            print(f"{step_name}: {output}")

        return self.results






    
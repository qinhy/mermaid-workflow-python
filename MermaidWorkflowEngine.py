# https://github.com/qinhy/mermaid-workflow-python

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

from pydantic import BaseModel, Field
from typing import List, Dict, DefaultDict
from collections import defaultdict

class GraphNode(BaseModel):
    prev: List[str] = Field(default_factory=list)
    next: List[str] = Field(default_factory=list)
    config: Dict[str, dict] = Field(default_factory=dict)
    maps: Dict[str, str] = Field(default_factory=dict)

Graph = DefaultDict[str, GraphNode]

def parse_mermaid(mermaid_text: str="""
graph TD
    LoadImage["{'para': {'path': './tmp/input.jpg'}}"]
    BlurImage["{'para': {'radius': 2}}"]
    BlurImage_2["{'para': {'radius': 5}}"]
    ResizeImage_01["{'para': {'width': 512, 'height': 512}}"]
    ResizeImage["{'para': {'width': 1024, 'height': 1024}}"]

    LoadImage -- "{'path':'path'}" --> ResizeImage_01
    ResizeImage_01 --> BlurImage
    BlurImage --> BlurImage_2
    BlurImage_2 --> ResizeImage
                  
%% ### 📌 Mermaid Graph Protocol (for beginners):
%% * `graph TD` → Start of a top-down Mermaid flowchart
%% * `NodeName[_optionalID]["{{...}}"]` (e.g., `ResizeImage_01`) → Define a node with initialization parameters in JSON-like format
%% * The initialization parameters **must not contain mapping information** — only raw valid values (e.g., numbers, strings, booleans)
%% * `A --> B` → Connect node A to node B (no field mapping)
%% * `A -- "{{'x':'y'}}" --> B` → Map output field `'x'` from A to input field `'y'` of B
%% * Use **valid field names** from each tool's input/output schema
"""):
    
    lines = [l.strip() for l in mermaid_text.strip().splitlines()]
    lines = [l for l in lines if ('["{' in l) or ('--' in l)]
    lines = [l for l in lines if '["{}"]' not in l]
    lines = [l for l in lines if not l.startswith('%%')]
    # graph = defaultdict(lambda: {"prev": [], "next": [], "config": {}, "maps": {}})
    graph: Graph = defaultdict(GraphNode)

    node_pattern = re.compile(r'^([\w-]+)\s*\[\s*"(.+)"\s*\]$')
    map_pattern = re.compile(r'^([\w-]+)\s*--\s*"(.*?)"\s*-->\s*([\w-]+)$')
    simple_pattern = re.compile(r'^([\w-]+)\s*-->\s*([\w-]+)$')

    def parse_json(s: str) -> Any:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception as e:
            print(s)
            raise e

    for l in lines:
        if not l or l.startswith("graph"):
            continue
        m = node_pattern.match(l)
        if m:
            node, cfg = m.groups()
            parsed = parse_json(cfg)
            if parsed is not None:
                graph[node].config = parsed
            continue
        m = map_pattern.match(l)
        if m:
            src, cfg, dst = m.groups()
            graph[src].next.append(dst)
            graph[dst].prev.append(src)
            parsed = parse_json(cfg)
            if parsed is not None:
                graph[src].maps[dst] = parsed
            continue
        m = simple_pattern.match(l)
        if m:
            src, dst = m.groups()
            graph[src].next.append(dst)
            graph[dst].prev.append(src)
            continue
        raise ValueError(f"Invalid Mermaid syntax: {l}")
    return dict(graph)

def validate_dep_single(needs: list[str], provided: list[str]) -> tuple[bool, list[str]]:
    missing = list(set(needs) - set(provided))
    if missing:
        print(f"❌ Missing required fields: {missing}")
        return False, missing

    # print("✅ All required fields are satisfied.")
    return True, []

def validate_dep_multi(needs: List[str], multi_provided: Dict[str, List[str]]) -> Tuple[bool, Dict[str, List[str]], List[str]]:
    # Flatten provided fields
    flat_provided = set()
    field_sources = defaultdict(list)

    for dep, fields in multi_provided.items():
        flat_provided.update(fields)
        for field in fields:
            if field in needs:  # Only track relevant fields
                field_sources[field].append(dep)

    # Reuse validate_dep_single
    is_valid, missing = validate_dep_single(needs, list(flat_provided))

    # Diagnostics
    for field, sources in field_sources.items():
        if len(sources) > 1:
            print(f"⚠️ Field '{field}' is provided by multiple sources: {sources}")
        else:
            print(f"✅ Field '{field}' is provided by: {sources[0]}")

    return is_valid, dict(field_sources), missing


class MermaidWorkflowFunctionTemplate(BaseModel):
    """Base class for workflow function nodes with parameters, arguments and returns."""
    
    class Parameter(BaseModel):
        """Static parameters that configure the function behavior."""
        pass

    class Arguments(BaseModel):
        """Input arguments received from predecessor nodes."""
        pass

    class Returness(BaseModel):
        """Output values passed to successor nodes."""
        pass

    para: Optional[Parameter|dict] = None
    args: Optional[Arguments|dict] = None
    rets: Optional[Returness|dict] = None
    run_at_init:bool = False

class MermaidWorkflowFunction(BaseModel):
    """Base class for workflow function nodes with parameters, arguments and returns."""
    
    class Parameter(BaseModel):
        """Static parameters that configure the function behavior."""
        pass

    class Arguments(BaseModel):
        """Input arguments received from predecessor nodes."""
        pass

    class Returness(BaseModel):
        """Output values passed to successor nodes."""
        pass

    # para: Optional[Parameter] = None
    # args: Optional[Arguments] = None
    # rets: Optional[Returness] = None
    run_at_init:bool = False

    def model_post_init(self,context):
        if self.run_at_init: self()

    def __call__(self) -> Returness:
        """Execute the workflow function and return results.
        
        Returness:
            Returness: Output values to be passed to successor nodes
        """
        raise NotImplementedError("Workflow functions must implement __call__")

    def update(self, data: dict) -> None:
        """Update para, args, and rets fields from the given data dict."""
        try:
            m:MermaidWorkflowFunctionTemplate = MermaidWorkflowFunctionTemplate.model_validate(data)
        except Exception as e:
            print(e)
            return

        if m.para and self.para:
            current = self.para.model_dump()
            current.update(**m.para.model_dump())
            self.para = self.Parameter(**current)

        if m.args and self.args:
            current = self.args.model_dump()
            current.update(**m.args.model_dump())
            self.args = self.Arguments(**current)

        if m.rets and self.rets:
            current = self.rets.model_dump()
            current.update(**m.rets.model_dump())
            self.rets = self.Returness(**current)
            
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
        if "Parameter" in defs:
            param_def = defs["Parameter"]
            Parameter = build_model_from_properties("Parameter", param_def["properties"], param_def.get("required", []))
            model_creation_args["para"] = (Parameter,...)

        if "Arguments" in defs:
            arg_def = defs["Arguments"]
            Arguments = build_model_from_properties("Arguments", arg_def["properties"], arg_def.get("required", []))
            model_creation_args["args"] = (Arguments,...)

        if "Returness" in defs:
            arg_def = defs["Returness"]
            Returness = build_model_from_properties("Returness", arg_def["properties"], arg_def.get("required", []))
            # model_creation_args["rets"] = (Returness,...)
            
        model_cls = create_model(name, **model_creation_args,__base__=MermaidWorkflowFunction)
        if "Parameter" in defs:model_cls.Parameter = Parameter
        if "Arguments" in defs:model_cls.Arguments = Arguments
        if "Returness" in defs:model_cls.Returness = Returness

        return model_cls

    @classmethod
    def get_para_class_name(cls) -> str:
        """Get the name of the Parameter class."""
        return cls.Parameter.__name__

    @classmethod
    def get_args_class_name(cls) -> str:
        """Get the name of the Arguments class."""
        return cls.Arguments.__name__

    @classmethod
    def get_rets_class_name(cls) -> str:
        """Get the name of the Returness class."""
        return cls.Returness.__name__ 

    @classmethod
    def get_model_fields(cls, model_name: str) -> list[str]:
        """Get field names from a specified model class.
        
        Args:
            model_name: Name of the model class ('Parameter', 'Arguments', or 'Returness')
            
        Returness:
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
        self.graph: dict[str, GraphNode] = {}
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
    
    def node_get(self, node_name: str):
        node_name = node_name.split('_')[0]
        cls:MermaidWorkflowFunction = self.name_to_class.get(node_name)
        return cls
    
    def node_args_fields(self, node_name: str):
        return self.node_get(node_name).get_argument_fields()

    def node_rets_fields(self, node_name: str):
        return self.node_get(node_name).get_return_fields()

    def validate_io(self) -> bool:
        print("\n🔍 Validating workflow I/O with mapping support...")

        unknown = set([n.split("_")[0] for n in list(self.graph.keys())]) - set(self.name_to_class)
        if unknown:
            print(f"❌ Unknown classes found: {unknown}")
            return False

        all_valid = True

        for node, meta in self.graph.items():
            deps = meta.prev
            node_cfg = meta.config
            required = set(self.node_args_fields(node))
            provided_fields = defaultdict(list)

            # No dependencies — check config directly
            if not deps:
                config_args = set(node_cfg.get('args', {}).keys())
                is_valid, missing = validate_dep_single(list(required), list(config_args))
                if not is_valid:
                    print(f"❌ Node '{node}' has no dependencies but requires inputs: {missing}")
                    all_valid = False
                else:
                    print(f"⚠️ Node '{node}' has no dependencies and no required inputs.")
                continue

            # Build provided fields from dependencies
            for dep in deps:
                dep_outputs = set(self.node_rets_fields(dep))
                dep_map = self.graph[dep].maps.get(node, {})

                # — 1. Validate explicit mappings
                bad_srcs = set(dep_map) - dep_outputs
                for src in bad_srcs:
                    print(f"❌ Field '{src}' not found in outputs of '{dep}'")
                    all_valid = False

                bad_dsts = set(dep_map.values()) - required
                for dst in bad_dsts:
                    print(f"⚠️ Mapping to '{dst}' ignored—it's not required by '{node}'")

                for src, dst in dep_map.items():
                    if src == dst and dst in required:
                        print(f"⚠️ Redundant explicit mapping '{src}→{dst}' for '{node}'")

                # — 2. Apply explicit mappings
                for src, dst in dep_map.items():
                    if src in dep_outputs and dst in required:
                        provided_fields[dst].append(dep)

                # — 3. Default 1:1 mappings
                unmapped_defaults = (dep_outputs & required) - set(dep_map.values())
                for field in unmapped_defaults:
                    provided_fields[field].append(dep)

                # — 4. Warn unused outputs
                used_outputs = set(dep_map.keys()).union(unmapped_defaults)
                unused = dep_outputs - used_outputs
                if unused:
                    print(f"⚠️ Outputs from '{dep}' to '{node}' never used: {sorted(unused)}")

            # — 5. Validate with validate_dep_multi
            multi_provided = defaultdict(list)
            for field, sources in provided_fields.items():
                for src in sources:
                    multi_provided[src].append(field)

            is_valid, field_sources, missing = validate_dep_multi(list(required), multi_provided)
            if not is_valid:
                for field in missing:
                    print(f"❌ Missing field '{field}' for node '{node}' from dependencies: {deps}")
                all_valid = False

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
        ts_graph = {node: meta.prev for node, meta in self.graph.items()}
        sorter = TopologicalSorter(ts_graph)
        execution_order = list(sorter.static_order())

        for node_name in execution_order:
            self.results[node_name] = {}
            cls:MermaidWorkflowFunction = self.node_get(node_name)

            # Collect inputs from dependencies
            args_data = {}
            deps = self.graph[node_name].prev
            for dep in deps:
                dep_results = self.results.get(dep, {})
                map_config:dict = self.graph[dep].maps
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
            conf = self.graph[node_name].config
            para_data = conf.get("para", {})
            args_data.update(conf.get("args", {}))

            cls_data = {}
            try:
                if hasattr(cls, "Parameter") and len(cls.Parameter.model_fields) > 0:
                    cls_data['para'] = cls.Parameter(**para_data)
                if hasattr(cls, "Arguments") and len(cls.Arguments.model_fields) > 0:
                    cls_data['args'] = cls.Arguments(**args_data)
            except Exception as e:
                print(f"❌ Error validating config for '{node_name}': {e}")
                raise e

            print(f"\n🔄 Executing node '{node_name}' with para: {para_data}")
            print(f"\n🔄 Executing node '{node_name}' with args: {args_data}")

            try:
                instance:MermaidWorkflowFunction = cls(**cls_data)
                cls_data['run_at_init'] = True
                res = ignite_func(instance, cls_data)
                print(f"\n🔄 Executing node '{node_name}' got res: {res}")

                # Extract return values
                if hasattr(instance, "rets") and hasattr(instance.rets, "model_dump"):
                    self.results[node_name] = instance.rets.model_dump()
                elif isinstance(res, dict) and "rets" in res:
                    self.results[node_name] = res["rets"]

            except Exception as e:
                print(f"❌ Error executing node '{node_name}':")
                print(e)
                print(cls.model_json_schema())
                print(cls_data)
                print(cls)
                raise e

        print("\n✅ Final outputs:")
        for step_name, output in self.results.items():
            print(f"{step_name}: {output}")

        return self.results






    
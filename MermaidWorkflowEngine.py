# https://github.com/qinhy/mermaid-workflow-python
import json
import re
import inspect
from collections import defaultdict
from graphlib import TopologicalSorter
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple, Type, Union
from pydantic import BaseModel, Field, create_model

logger = print
# logger = lambda *args,**kwargs: None

# ------------------------------
# Mermaid Protocol Documentation
# ------------------------------

def mermaid_protocol() -> str:
    return '''
### 📌 Mermaid Graph Protocol (for beginners):
* `graph TD` → Start of a top-down Mermaid flowchart
* `NodeName[_optionalID]["{{...}}"]` (e.g., `ResizeImage_01`) → Define a node with initialization parameters in JSON-like format
* The initialization parameters **must not contain mapping information** — only raw valid values (e.g., numbers, strings, booleans)
* `A --> B` → Connect node A to node B (no field mapping)
* `A -- "{{'x':'y'}}" --> B` → Map output field `'x'` from A to input field `'y'` of B
* Use **valid field names** from each tool's input/output schema
'''

# ------------------------------
# Data Structures
# ------------------------------

class GraphNode(BaseModel):
    prev: List[str] = Field(default_factory=list)
    next: List[str] = Field(default_factory=list)
    config: Dict[str, dict] = Field(default_factory=dict)
    maps: List[Tuple[str, str]] = Field(default_factory=list)  # (source_field, destination_field)

Graph = DefaultDict[str, GraphNode]

# ------------------------------
# Parser
# ------------------------------

def parse_mermaid(mermaid_text: str = """
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
""") -> Dict[str, GraphNode]:
# Got
# {'LoadImage': GraphNode(prev=[], next=['ResizeImage_01'], config={'para': {'path': './tmp/input.jpg'}}, 
#                                   maps=[('LoadImage::path', 'ResizeImage_01::path')]),
#  'BlurImage': GraphNode(prev=['ResizeImage_01'], next=['BlurImage_2'], config={'para': {'radius': 2}}, maps=[]),
#  'BlurImage_2': GraphNode(prev=['BlurImage'], next=['ResizeImage'], config={'para': {'radius': 5}}, maps=[]),
#  'ResizeImage_01': GraphNode(prev=['LoadImage'], next=['BlurImage'], config={'para': {'width': 512, 'height': 512}}, maps=[]),
#  'ResizeImage': GraphNode(prev=['BlurImage_2'], next=[], config={'para': {'width': 1024, 'height': 1024}}, maps=[])}

    lines = [l.strip() for l in mermaid_text.strip().splitlines()]
    lines = [l for l in lines if ('["{' in l) or ('--' in l)]
    lines = [l for l in lines if '["{}"]' not in l and not l.startswith('%%')]

    graph: Graph = defaultdict(GraphNode)

    node_pattern = re.compile(r'^([\w-]+)\s*\[\s*"(.+)"\s*\]$')
    map_pattern = re.compile(r'^([\w-]+)\s*--\s*"(.*?)"\s*-->\s*([\w-]+)$')
    simple_pattern = re.compile(r'^([\w-]+)\s*-->\s*([\w-]+)$')

    def parse_json(s: str) -> Any:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception as e:
            raise ValueError(f"❌ Error parsing JSON content: {s} → {e}")

    for l in lines:
        if not l or l.startswith("graph"):
            continue

        # Node definition
        m = node_pattern.match(l)
        if m:
            node, cfg = m.groups()
            parsed = parse_json(cfg)
            if parsed is not None:
                graph[node].config = parsed
            continue

        # Mapped edge
        m = map_pattern.match(l)
        if m:
            src, cfg, dst = m.groups()
            graph[src].next.append(dst)
            graph[dst].prev.append(src)
            parsed = parse_json(cfg)
            if parsed is not None:
                for src_field, dst_field in parsed.items():
                    graph[src].maps.append((f"{src}::{src_field}", f"{dst}::{dst_field}"))
            continue

        # Simple edge
        m = simple_pattern.match(l)
        if m:
            src, dst = m.groups()
            graph[src].next.append(dst)
            graph[dst].prev.append(src)
            continue

        raise ValueError(f"❌ Invalid Mermaid syntax: {l}")

    return dict(graph)

def parse_mermaid_old(mermaid_text: str="""
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
            logger(f"❌ Error parsing JSON: {e}")
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
        logger(f"❌ Missing required fields: {missing}")
        return False, missing

    # logger("✅ All required fields are satisfied.")
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
            logger(f"⚠️ Field '{field}' is provided by multiple sources: {sources}")
        else:
            logger(f"✅ Field '{field}' is provided by: {sources[0]}")

    return is_valid, dict(field_sources), missing


class MermaidWorkflowFunctionTemplate(BaseModel):
    para: Optional[dict] = Field(None,description="Parameter")
    args: Optional[dict] = Field(None,description="Arguments")
    rets: Optional[dict] = Field(None,description="Returness")
    
    @staticmethod
    def keys():
        return list(MermaidWorkflowFunctionTemplate.model_fields.keys())
    
    @staticmethod
    def values():
        return list(map(lambda x:x.description, MermaidWorkflowFunctionTemplate.model_fields.values()))

class MermaidWorkflowFunction(BaseModel):
    """Base class for workflow function nodes with parameters, arguments and returns."""
    
    # class Parameter(BaseModel):
    #     """Static parameters that configure the function behavior."""
    #     pass

    # class Arguments(BaseModel):
    #     """Input arguments received for function behavior."""
    #     pass

    # class Returness(BaseModel):
    #     """Output values of function."""
    #     pass

    # should not define just for comment
    # para: Optional[Parameter|dict] = None
    # args: Optional[Arguments|dict] = None
    # rets: Optional[Returness|dict] = None

    run_at_init:bool = False

    def model_post_init(self,context):
        if self.run_at_init: self()

    # def __call__(self) -> Returness:
    #     raise NotImplementedError("Workflow functions must implement __call__")

    def update(self, data: dict) -> None:
        """Update para, args, and rets fields from the given data dict."""
        try:
            m:MermaidWorkflowFunctionTemplate = MermaidWorkflowFunctionTemplate.model_validate(data)
        except Exception as e:
            logger(f"❌ Error validating data: {e}")
            return

        if m.para and self.para:
            current = self.para.model_dump()
            current.update(**m.para)
            self.para = self.Parameter(**current)

        if m.args and self.args:
            current = self.args.model_dump()
            current.update(**m.args)
            self.args = self.Arguments(**current)

        if m.rets and self.rets:
            current = self.rets.model_dump()
            current.update(**m.rets)
            self.rets = self.Returness(**current)

    @classmethod
    def from_json_schema(cls, schema: Dict[str, Any]) -> Type[BaseModel]:
        def resolve_ref(schema: Dict[str, Any], ref: str) -> Dict[str, Any]:
            """Resolve local JSON Pointer like '#/$defs/Order/$defs/Item'."""
            if not ref.startswith("#/"):
                raise ValueError(f"Only local refs are supported: {ref}")
            parts = ref.lstrip("#/").split("/")
            result = schema
            for part in parts:
                if part not in result:
                    raise KeyError(f"Could not resolve ref path '{ref}' at '{part}'")
                result = result[part]
            return result

        def json_schema_type_to_python(json_type: str) -> Any:
            return {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "object": Dict[str, Any],
                "array": List[Any],
                "null": type(None),
                "any": Any
            }.get(json_type, Any)

        def build_model_from_properties(
            name: str,
            props: Dict[str, dict],
            required: List[str],
            root_schema: Dict[str, Any],
            known_models: Dict[str, Type[BaseModel]]
        ) -> Type[BaseModel]:
            fields: Dict[str, Tuple[Any, Any]] = {}
            for key, spec in props.items():
                # Handle $ref
                if "$ref" in spec:
                    ref = spec["$ref"]
                    ref_schema = resolve_ref(root_schema, ref)
                    ref_name = "_".join(ref.strip("#/").split("/"))  # unique model name

                    if ref_name not in known_models:
                        submodel = build_model_from_properties(
                            ref_name,
                            ref_schema.get("properties", {}),
                            ref_schema.get("required", []),
                            root_schema,
                            known_models
                        )
                        known_models[ref_name] = submodel

                    field_type = known_models[ref_name]
                    if key not in required:
                        field_type = Optional[field_type]
                    fields[key] = (field_type, ... if key in required else None)
                    continue

                # Handle anyOf (union types)
                if "anyOf" in spec:
                    types = []
                    for type_spec in spec["anyOf"]:
                        if "$ref" in type_spec:
                            ref = type_spec["$ref"]
                            ref_schema = resolve_ref(root_schema, ref)
                            ref_name = "_".join(ref.strip("#/").split("/"))
                            if ref_name not in known_models:
                                known_models[ref_name] = build_model_from_properties(
                                    ref_name,
                                    ref_schema.get("properties", {}),
                                    ref_schema.get("required", []),
                                    root_schema,
                                    known_models
                                )
                            types.append(known_models[ref_name])
                        else:
                            json_type = type_spec.get("type")
                            types.append(json_schema_type_to_python(json_type))
                    field_type = Union[tuple(types)]
                    if key not in required:
                        field_type = Optional[field_type]
                    fields[key] = (field_type, ... if key in required else None)
                    continue

                # Inline object
                if spec.get("type") == "object" and "properties" in spec:
                    submodel_name = f"{name}_{key.capitalize()}"
                    submodel = build_model_from_properties(
                        submodel_name,
                        spec["properties"],
                        spec.get("required", []),
                        root_schema,
                        known_models
                    )
                    known_models[submodel_name] = submodel
                    field_type = submodel
                    if key not in required:
                        field_type = Optional[field_type]
                    fields[key] = (field_type, ... if key in required else None)
                    continue

                # Array type
                if spec.get("type") == "array":
                    items = spec.get("items", {})
                    if "$ref" in items:
                        item_ref:str = items["$ref"]
                        ref_schema = resolve_ref(root_schema, item_ref)
                        ref_name = "_".join(item_ref.strip("#/").split("/"))
                        if ref_name not in known_models:
                            known_models[ref_name] = build_model_from_properties(
                                ref_name,
                                ref_schema.get("properties", {}),
                                ref_schema.get("required", []),
                                root_schema,
                                known_models
                            )
                        item_type = known_models[ref_name]
                    else:
                        item_type = json_schema_type_to_python(items.get("type", "any"))
                    field_type = List[item_type]
                    if key not in required:
                        field_type = Optional[field_type]
                    fields[key] = (field_type, ... if key in required else None)
                    continue

                # Primitive types
                json_type = spec.get("type", "any")
                field_type = json_schema_type_to_python(json_type)
                if key not in required:
                    field_type = Optional[field_type]
                fields[key] = (field_type, ... if key in required else None)

            return create_model(name, __module__=__name__, **fields, __base__=MermaidWorkflowFunction)

        known_models: Dict[str, Type[BaseModel]] = {}
        model_name = schema.get('title', 'GeneratedModel')
        return build_model_from_properties(
            model_name,
            schema.get("properties", {}),
            schema.get("required", []),
            schema,
            known_models
        )
    
    @classmethod
    def from_mcp(cls, mcp_single_func_data: dict) -> 'MermaidWorkflowFunction':
        defs = mcp_single_func_data["inputSchema"]
        defs['title'] = mcp_single_func_data['name']
        return MermaidWorkflowFunction.from_json_schema(defs)

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
        model_cls:Type[BaseModel] = getattr(cls, model_name)
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
    
class MermaidWorkflowEngine(BaseModel):
    protocol:str = Field(mermaid_protocol(),description='mermaid workflow protocol')
    mermaid_text:str = Field('',description='mermaid workflow text')

    _name_to_class: Dict[str, Any] = {}
    _graph: dict[str, GraphNode] = {}
    _results: Dict[str, Any] = {}

    def model_register(self, model_registry: Dict[Type, str])->'MermaidWorkflowEngine':
        self.mermaid_text = ""
        model_registry = [(k,v) if type(k) is str else (v,k) for k, v in model_registry.items()]
        self._name_to_class = {k:v for k,v in model_registry}
        
        for k,v in model_registry:
            if type(v) is dict:
                self._name_to_class[k] = MermaidWorkflowFunction.from_mcp(v)
        return self

    def extract_mermaid_text(self, text: str) -> str:
        """Extract Mermaid flowchart text from a given text."""
        mermaid_pattern = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)
        match = mermaid_pattern.search(text)
        if match:
            return match.group(1).strip()
        return ""

    def parse_mermaid(self, mermaid_text:str) -> Dict[str, Dict[str, Any]]:
        return parse_mermaid(mermaid_text)
    
    def node_get(self, node_name: str, with_instance=False):
        node_name = node_name.split('_')[0]
        instance = None
        tmp = self._name_to_class.get(node_name)
        if tmp is None:
            raise ValueError(f'unknow node {node_name}')
        if type(tmp) is tuple:
            cls:Type[MermaidWorkflowFunction] = tmp[0]
            instance = tmp[1]
        else:
            cls = tmp
        if with_instance:
            return cls,instance
        return cls
    
    @staticmethod
    def create_Arguments_model_by__call__(instance):
        signature = inspect.signature(instance.__call__)
        params = signature.parameters
        # Convert signature parameters to fields for Pydantic
        fields = {}
        for name, param in params.items():
            if name == 'self':continue
            annotation = param.annotation
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[name] = (annotation, default)
        # Create the Pydantic model dynamically
        Arguments = create_model("Arguments", **fields)
        return Arguments
    
    @staticmethod
    def create_Returness_model_by__call__(instance):
        signature = inspect.signature(instance.__call__)
        return_annotation = signature.return_annotation
        fields = {}
        fields['data'] = (return_annotation, ...)
        Returness = create_model("Returness", **fields)
        return Returness

    def get_fields(self, node:Type[BaseModel],target:str='args',required=True)->set[str]:

        if target in node.model_fields:
            target_annotation = node.model_fields[target].annotation
            if hasattr(target_annotation,'__args__'):
                target_item_type:Type[BaseModel] = target_annotation.__args__[0]
            elif hasattr(target_annotation,'model_fields'):
                target_item_type:Type[BaseModel] = target_annotation
            else:
                raise ValueError(f'unknow type of {target_annotation}')
            
        elif hasattr(node,'__call__'):
            if target == 'args':
                target_item_type = self.create_Arguments_model_by__call__(node)
            elif target == 'rets':
                target_item_type = self.create_Returness_model_by__call__(node)
        else:
            raise ValueError(f'unknow type of {node}')
        
        if required:
            return set([
                name for name, field in target_item_type.model_fields.items()
                if field.is_required()
            ])
        else:
            return set([
                name for name, _ in target_item_type.model_fields.items()
            ])
        
    def validate_io(self, initial_args: Dict[str, Any] = {}) -> bool:
        logger("🔍 Validating workflow I/O with mapping support...")

        unknown_classes = {
            n.split("_")[0] for n in self._graph.keys()
        } - set(self._name_to_class)
        if unknown_classes:
            logger(f"❌ Unknown classes found: {unknown_classes}")
            return False

        all_valid = True

        for node_name, meta in self._graph.items():
            deps = meta.prev
            node_cfg = meta.config
            node = self.node_get(node_name)
            required_fields = self.get_fields(node, 'args', required=True)
            required = {f'{node_name}::{f}' for f in required_fields}
            provided_fields = defaultdict(list)

            if not deps:
                # Combine config and initial_args for validation
                config_args = set(node_cfg.get('args', {}).keys()) | set(initial_args.keys())
                is_valid, missing = validate_dep_single(list(required_fields), list(config_args))
                if not is_valid:
                    logger(f"❌ Node '{node_name}' has no dependencies but requires inputs: {missing}")
                    all_valid = False
                else:
                    logger(f"✅ Node '{node_name}' with no dependencies passed input validation.")
                continue

            for dep in deps:
                dep_node = self.node_get(dep)
                dep_outputs = {
                    f'{dep}::{f}' for f in self.get_fields(dep_node, 'rets', required=False)
                }

                mappings = self._graph[dep].maps
                explicit_srcs = [m[0] for m in mappings]
                explicit_dsts = [m[1] for m in mappings]

                # Validate mappings
                for src in set(explicit_srcs) - dep_outputs:
                    logger(f"❌ Field '{src}' not in outputs of '{dep}' (has {[*dep_outputs]})")
                    all_valid = False

                for dst in set(explicit_dsts) - required:
                    logger(f"⚠️ Mapping to '{dst}' ignored — not required by '{node_name}'")

                for src, dst in zip(explicit_srcs, explicit_dsts):
                    if src == dst and dst in required:
                        logger(f"⚠️ Redundant mapping '{src}→{dst}' for '{node_name}'")

                # Apply explicit mappings
                for src, dst in zip(explicit_srcs, explicit_dsts):
                    provided_fields[dst].append(src)

                # Implicit match
                unmapped_outputs = dep_outputs - set(explicit_srcs)
                for output in unmapped_outputs:
                    field = output.split("::", 1)[1]
                    candidate_dst = f'{node_name}::{field}'
                    if candidate_dst in required:
                        provided_fields[candidate_dst].append(output)

            # Final validation
            missing = required - provided_fields.keys()
            if missing:
                logger(f"❌ Node '{node_name}' is missing required inputs: {missing}")
                all_valid = False
            else:
                logger(f"✅ Node '{node_name}' I/O validated successfully.")

        return all_valid

    def run(self, mermaid_text: str = None, ignite_func: Optional[Callable] = None,
            initial_args: dict = {}) -> Dict[str, Any]:
        
        ignite_func = ignite_func or (lambda obj, args: obj())

        if mermaid_text:
            self._graph = self.parse_mermaid(mermaid_text)

        if not self.validate_io(initial_args=initial_args):
            logger("❌ Workflow validation failed. Exiting.")
            return {}

        # Topological order
        ts_graph = {node: meta.prev for node, meta in self._graph.items()}
        execution_order = list(TopologicalSorter(ts_graph).static_order())

        for i, node_name in enumerate(execution_order):
            self._results[node_name] = {}
            meta = self._graph[node_name]
            deps = meta.prev
            cls, instance = self.node_get(node_name, with_instance=True)
            cls: Type[MermaidWorkflowFunction]

            # Collect inputs from deps
            args_data = {}
            required_args = self.get_fields(cls, 'args', required=True)

            for dep in deps:
                dep_result = self._results.get(dep, {})
                dep_mappings = [m for m in self._graph[dep].maps if node_name in m[1]]

                # Apply mappings
                for src, dst in dep_mappings:
                    src, dst = src.split("::", 1)[1], dst.split("::", 1)[1]
                    if src in dep_result:
                        args_data[dst] = dep_result[src]                

                # Implicit match
                for field in dep_result:
                    if field in required_args and field not in args_data:
                        args_data[field] = dep_result[field]

            # Merge static config
            config = meta.config
            para_data = config.get("para", {})
            args_data.update(config.get("args", {}))
            
            if i == 0 and initial_args:
                args_data.update(initial_args)
            

            cls_data = {}
            if para_data:
                cls_data['para'] = para_data
            if args_data:
                cls_data['args'] = args_data

            try:
                logger(f"🔄 Executing node '{node_name}' with: {cls_data}")
                if instance is None:
                    instance = cls(**cls_data)

                cls_data['run_at_init'] = True
                res = ignite_func(instance, cls_data)

                logger(f"✅ Node '{node_name}' executed successfully: {res}")

                # Handle outputs
                if hasattr(instance, "rets") and hasattr(instance.rets, "model_dump"):
                    self._results[node_name] = instance.rets.model_dump()
                elif isinstance(res, dict) and "rets" in res:
                    self._results[node_name] = res["rets"]
                elif hasattr(instance, "__call__"):
                    Returness: Type[BaseModel] = self.create_Returness_model_by__call__(instance)
                    self._results[node_name] = Returness(data=res).model_dump()
                else:
                    raise ValueError(f"Node '{node_name}' must return dict with 'rets' or have a 'rets' attribute.")

            except Exception as e:
                logger(f"❌ Error executing node '{node_name}': {e}")
                raise

        self._results['final'] = self._results.get(node_name, {})
        logger(f"✅ Final outputs:\n{json.dumps(self._results, indent=4)}")
        return self._results

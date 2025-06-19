# https://github.com/qinhy/mermaid-workflow-python

import json
import re
from collections import defaultdict
from graphlib import TopologicalSorter
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel, Field, create_model

logger = print
# logger = lambda *args,**kwargs: None

class GraphNode(BaseModel):
    prev: List[str] = Field(default_factory=list)
    next: List[str] = Field(default_factory=list)
    config: Dict[str, dict] = Field(default_factory=dict)
    maps: Dict[str, dict] = Field(default_factory=dict)

Graph = DefaultDict[str, GraphNode]

def mermaid_protocol()->str:
    return '''
### 📌 Mermaid Graph Protocol (for beginners):
* `graph TD` → Start of a top-down Mermaid flowchart
* `NodeName[_optionalID]["{{...}}"]` (e.g., `ResizeImage_01`) → Define a node with initialization parameters in JSON-like format
* The initialization parameters **must not contain mapping information** — only raw valid values (e.g., numbers, strings, booleans)
* `A --> B` → Connect node A to node B (no field mapping)
* `A -- "{{'x':'y'}}" --> B` → Map output field `'x'` from A to input field `'y'` of B
* Use **valid field names** from each tool's input/output schema
'''

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
    
    class Parameter(BaseModel):
        """Static parameters that configure the function behavior."""
        pass

    class Arguments(BaseModel):
        """Input arguments received for function behavior."""
        pass

    class Returness(BaseModel):
        """Output values of function."""
        pass

    # should not define just for comment
    para: Optional[Parameter|dict] = None
    args: Optional[Arguments|dict] = None
    rets: Optional[Returness|dict] = None

    run_at_init:bool = False

    def model_post_init(self,context):
        if self.run_at_init: self()

    def __call__(self) -> Returness:
        raise NotImplementedError("Workflow functions must implement __call__")

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
    
    def _collect_args_from_deps(self, node_name: str, args_fields: List[str], deps: List[str]) -> Dict[str, Any]:
        args_data = {}
        used_fields = set()
        field_source_map = defaultdict(list)

        for dep in deps:
            dep_result = self._results.get(dep, {})
            for key in dep_result:
                field_source_map[dep].append(key)

        for field in args_fields:
            found = False
            for dep in deps:
                dep_result = self._results.get(dep, {})
                if field in dep_result:
                    args_data[field] = dep_result[field]
                    used_fields.add(field)
                    found = True
                    break
            if not found:
                logger(f"⚠️ Warning: Field '{field}' required by '{node_name}' not found in dependencies: {deps}")

        for dep, fields in field_source_map.items():
            unused = set(fields) - used_fields
            if unused:
                logger(f"ℹ️ Info: Fields from '{dep}' not used by '{node_name}': {sorted(unused)}")

        return args_data
    
    def node_get(self, node_name: str):
        node_name = node_name.split('_')[0]
        cls:Type[MermaidWorkflowFunction] = self._name_to_class.get(node_name)
        return cls
    
    def node_args_fields(self, node_name: str):
        return self.node_get(node_name).get_argument_fields()

    def node_rets_fields(self, node_name: str):
        return self.node_get(node_name).get_return_fields()

    def validate_io(self) -> bool:
        logger("🔍 Validating workflow I/O with mapping support...")

        unknown = set([n.split("_")[0] for n in list(self._graph.keys())]) - set(self._name_to_class)
        if unknown:
            logger(f"❌ Unknown classes found: {unknown}")
            return False

        all_valid = True

        def get_fields(node:Type[BaseModel],target:str='args',required=True):
            target_annotation = node.model_fields[target].annotation
            if hasattr(target_annotation,'__args__'):
                target_item_type:Type[BaseModel] = target_annotation.__args__[0]
            elif hasattr(target_annotation,'model_fields'):
                target_item_type:Type[BaseModel] = target_annotation
            else:
                raise ValueError(f'unknow type of {target_annotation}')
            if required:
                return set([
                    name for name, field in target_item_type.model_fields.items()
                    if field.is_required()
                ])
            else:
                return set([
                    name for name, field in target_item_type.model_fields.items()
                ])


        for node_name, meta in self._graph.items():
            deps = meta.prev
            node_cfg = meta.config
            node = self.node_get(node_name)
            required = get_fields(node,'args',required=True)
            provided_fields = defaultdict(list)

            # No dependencies — check config directly
            if not deps:
                config_args = set(node_cfg.get('args', {}).keys())
                is_valid, missing = validate_dep_single(list(required), list(config_args))
                if not is_valid:
                    logger(f"❌ Node '{node_name}' has no dependencies but requires inputs: {missing}")
                    all_valid = False
                else:
                    logger(f"⚠️ Node '{node_name}' has no dependencies and no required inputs.")
                continue

            # Build provided fields from dependencies
            for dep in deps:
                dep_node = self.node_get(dep)          
                dep_outputs = get_fields(dep_node,'rets',required=False)
                dep_map = self._graph[dep].maps.get(node_name, {})

                # — 1. Validate explicit mappings
                bad_srcs = set(dep_map) - dep_outputs
                for src in bad_srcs:
                    logger(f"❌ Field '{src}' not found in outputs of '{dep}'(has {[*dep_outputs]})")
                    all_valid = False

                bad_dsts = set(dep_map.values()) - required
                for dst in bad_dsts:
                    logger(f"⚠️ Mapping to '{dst}' ignored—it's not required by '{node_name}'")

                for src, dst in dep_map.items():
                    if src == dst and dst in required:
                        logger(f"⚠️ Redundant explicit mapping '{src}→{dst}' for '{node_name}'")

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
                    logger(f"⚠️ Outputs from '{dep}' to '{node_name}' never used: {sorted(unused)}")

            # — 5. Validate with validate_dep_multi
            multi_provided = defaultdict(list)
            for field, sources in provided_fields.items():
                for src in sources:
                    multi_provided[src].append(field)

            is_valid, field_sources, missing = validate_dep_multi(list(required), multi_provided)
            if not is_valid:
                for field in missing:
                    logger(f"❌ Missing field '{field}' for node '{node_name}' from dependencies: {deps}")
                all_valid = False

        if all_valid:
            logger("✅ Workflow validation passed: All inputs are satisfied.")
        else:
            logger("❌ Workflow validation failed. See messages above.")

        return all_valid

    def run(self, mermaid_text: str = None, ignite_func: Optional[Callable] = None) -> Dict[str, Any]:
        if ignite_func is None:
            ignite_func = lambda obj, args: obj()
        if mermaid_text is not None:
            self._graph = self.parse_mermaid(mermaid_text)
        if not self.validate_io():
            logger("❌ Workflow validation failed. Exiting.")
            return {}

        # Build the dependency graph for topological sorting
        ts_graph = {node: meta.prev for node, meta in self._graph.items()}
        sorter = TopologicalSorter(ts_graph)
        execution_order = list(sorter.static_order())

        for node_name in execution_order:
            self._results[node_name] = {}
            cls:Type[MermaidWorkflowFunction] = self.node_get(node_name)

            # Collect inputs from dependencies
            args_data = {}
            deps = self._graph[node_name].prev
            for dep in deps:
                dep_results = self._results.get(dep, {})
                map_config:dict = self._graph[dep].maps
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
            conf = self._graph[node_name].config
            para_data = conf.get("para", {})
            args_data.update(conf.get("args", {}))

            cls_data = {}
            try:
                if 'para' in cls.model_fields:
                    cls_data['para'] = {**para_data}
                if 'args' in cls.model_fields:
                    cls_data['args'] = {**args_data}
            except Exception as e:
                logger(f"❌ Error validating config for '{node_name}': {e}")
                raise e

            logger(f"🔄 Executing node '{node_name}' with : {cls_data}")
            try:
                instance:MermaidWorkflowFunction = cls(**cls_data)
                cls_data['run_at_init'] = True
                res = ignite_func(instance, cls_data)
                logger(f"✅ Executing node '{node_name}' got res: {res}")

                # Extract return values
                if hasattr(instance, "rets") and hasattr(instance.rets, "model_dump"):
                    self._results[node_name] = instance.rets.model_dump()
                elif isinstance(res, dict) and "rets" in res:
                    self._results[node_name] = res["rets"]

            except Exception as e:
                logger(f"❌ Error executing node '{node_name}':")
                logger(e)
                raise e

        self._results['final'] = self._results[node_name]
        logger(f"✅ Final outputs:\n{json.dumps(self._results,indent=4)}")
        return self._results








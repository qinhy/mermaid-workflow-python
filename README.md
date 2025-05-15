
# Mermaid Workflow Python

A Python library for defining, visualizing, and executing workflows using Mermaid flowchart syntax with strong typing support. This library allows you to create complex data processing pipelines with clear input/output contracts between workflow steps.

## Overview

Mermaid Workflow Python provides a framework for:

1. Defining workflows as Mermaid flowcharts
2. Enforcing type safety with Pydantic models
3. Automatic dependency resolution and execution order
4. Data mapping between workflow steps
5. Validation of workflow input/output contracts

## Features

- **Mermaid-based Workflow Definition**: Define workflows using the popular Mermaid flowchart syntax
- **Strong Typing**: Use Pydantic models to define input/output contracts for each workflow step
- **Automatic Dependency Resolution**: Topological sorting ensures steps execute in the correct order
- **Data Mapping**: Flexible mapping of outputs from one step to inputs of another
- **Validation**: Automatic validation of workflow structure and data contracts
- **OOP and Functional Approaches**: Support for both object-oriented and functional programming styles

## Installation

```bash
# Clone the repository
git clone https://github.com/qinhy/mermaid-workflow-python.git
cd mermaid-workflow-python
```

## Usage

### Simple Example

Here's a basic example of defining and running a workflow:

```python
# Define workflow steps with typed signatures
mermaid_definition = """
graph TD
    A["()->{x:int}"] --> B
    B["(x:int, y:int)->{z:int}"] --> C
    C["(z:int)->{}"]
    D["()->{y:int}"] --> B
"""

# Implement the workflow functions
def A():
    return {"x": 10}

def D():
    return {"y": 32}

def B(x, y):
    return {"z": x + y}

def C(z):
    print("Final result:", z)
    return {}

# Map node names to functions
node_functions = {
    'A': A,
    'B': B,
    'C': C,
    'D': D
}

# Run the workflow
run_workflow(graph, metadata, start_nodes)
```

### OOP Approach with Pydantic Models

For more complex workflows, you can use an object-oriented approach with Pydantic models:

```python
from pydantic import BaseModel
from typing import List

class LoadUserData:
    class Pars(BaseModel):
        endpoint: str = "default_endpoint"

    class Args(BaseModel):
        pass

    class Rets(BaseModel):
        user_id: int
        name: str

    pars: Pars | None = None
    args: Args | None = None
    rets: Rets | None = None

    def __call__(self) -> 'LoadUserData.Rets':
        # Implementation
        self.rets = self.Rets(user_id=1, name="Alice")
        return self.rets

# Define workflow
mermaid_definition = """
graph TD
    LoadUserData["{'pars':{'endpoint':'custom_endpoint'}}"] --> NextStep
    ...
"""

# Create registry
model_registry = {
    LoadUserData: "LoadUserData",
    # Other workflow steps
}

# Initialize and run the workflow engine
engine = MermaidWorkflowEngine(mermaid_definition, model_registry)
results = engine.run()
```

### Using as a Library

You can also use the `MermaidWorkflowEngine` class directly:

```python
from MermaidWorkflowEngine import MermaidWorkflowEngine, MermaidWorkflowFunction

# Define your workflow steps as subclasses of MermaidWorkflowFunction
class MyStep(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        # Parameters definition
        pass
        
    class Arguments(BaseModel):
        # Input arguments definition
        pass
        
    class Returns(BaseModel):
        # Output values definition
        pass
        
    def __call__(self) -> Returns:
        # Implementation
        pass

# Create registry and run workflow
engine = MermaidWorkflowEngine({
    MyStep: "MyStep",
    # Other steps
})

results = engine.run(mermaid_text)
```

## Advanced Features

### Data Mapping

You can explicitly map outputs from one step to inputs of another using the Mermaid syntax:

```
StepA -- "{'output_field': 'input_field'}" --> StepB
```

### Workflow Validation

The engine validates that all required inputs for each step are provided by its dependencies:

```python
engine = MermaidWorkflowEngine(mermaid_text, model_registry)
if engine.validate_io():
    results = engine.run()
```

## Examples

The repository includes several examples demonstrating different features:

1. `1.simple.py` - Basic workflow with function signatures
2. `2.with_args.py` - Workflow with arguments
3. `3.model_argstest3.py` - Using Pydantic models for type safety
4. `4.complex_flow.py` - More complex workflow example
5. `5.OOP_complex_flow.py` - Object-oriented approach for complex workflows
6. `6.as_lib.py` - Using the library in your own projects
7. `7.use_lib.py` - Example of importing and using the library
8. `8.test_lib.py` - Testing the library

## License
MIT
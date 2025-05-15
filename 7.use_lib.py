from pydantic import BaseModel
from MermaidWorkflowEngine import MermaidWorkflowEngine, MermaidWorkflowFunction

# -------- Mermaid definition --------
mermaid_definition = """
graph TD
    Start["{'para':{'a':10, 'b':-1}}"]
    Multiply["{'para':{'factor':10}}"]
    
    Start --> AddNumbers
    AddNumbers --> Multiply
    Multiply --> End
"""

# -------- Pydantic Models --------
class Start(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        a: int = 3
        b: int = 4

    class Returns(BaseModel):
        a: int
        b: int

    para: Parameters = Parameters()
    rets: Returns | None = None

    def __call__(self) -> Returns:
        print("Start: Providing numbers")
        self.rets = self.Returns(**self.para.model_dump())
        return self.rets
        
class AddNumbers(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        a: int
        b: int

    class Returns(BaseModel):
        value: int  # Changed from sum -> value

    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self) -> Returns:
        print("AddNumbers: Adding", self.args.a, "+", self.args.b)
        self.rets = self.Returns(value=self.args.a + self.args.b)
        return self.rets
        
class Multiply(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        factor: int = 2  # Default multiply factor

    class Arguments(BaseModel):
        value: int

    class Returns(BaseModel):
        value: int

    para: Parameters = Parameters()
    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self) -> Returns:
        factor = self.para.factor if self.para else 2
        print(f"Multiply: Multiplying {self.args.value} * {factor}")
        self.rets = self.Returns(value=self.args.value * factor)
        return self.rets

        
class End(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        value: int  # This now matches Multiply's output

    args: Arguments | None = None

    def __call__(self):
        print("End: Final result is", self.args.value)


# -------- Function and model registry --------

model_registry = {
    Start:'Start',
    AddNumbers:'AddNumbers',
    Multiply:'Multiply',
    End:'End',
}

# -------- Main --------
if __name__ == "__main__":
    engine = MermaidWorkflowEngine(mermaid_definition, model_registry)

    # Optional validation
    if engine.validate_io():
        results = engine.run()
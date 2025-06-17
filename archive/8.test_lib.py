from pydantic import BaseModel
from MermaidWorkflowEngine import MermaidWorkflowEngine, MermaidWorkflowFunction

class Start(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        x: int
        y: int
        z: int

    class Returns(BaseModel):
        x: int
        y: int
        z: int

    para: Parameters = Parameters(x=10, y=5, z=3)
    rets: Returns | None = None

    def __call__(self):
        print("Start: Inputs ->", self.para)
        self.rets = self.Returns(**self.para.model_dump())
        return self.rets
        
class Add(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        a: int
        b: int

    class Returns(BaseModel):
        sum: int

    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self):
        result = self.args.a + self.args.b
        print(f"Add: {self.args.a} + {self.args.b} = {result}")
        self.rets = self.Returns(sum=result)
        return self.rets
        
class Subtract(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        a: int
        b: int

    class Returns(BaseModel):
        result: int

    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self):
        result = self.args.a - self.args.b
        print(f"Subtract: {self.args.a} - {self.args.b} = {result}")
        self.rets = self.Returns(result=result)
        return self.rets
        
class Multiply(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        factor: int = 2

    class Arguments(BaseModel):
        x: int

    class Returns(BaseModel):
        product: int

    para: Parameters = Parameters(factor=4)
    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self):
        product = self.args.x * self.para.factor
        print(f"Multiply: {self.args.x} * {self.para.factor} = {product}")
        self.rets = self.Returns(product=product)
        return self.rets
        
class ValidateResult(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        product: int

    class Returns(BaseModel):
        valid: bool

    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self):
        is_valid = self.args.product > 0
        print(f"ValidateResult: {self.args.product} > 0 = {is_valid}")
        self.rets = self.Returns(valid=is_valid)
        return self.rets
        

class Divide(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        numerator: int
        denominator: int

    class Returns(BaseModel):
        quotient: int | None

    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self):
        if self.args.denominator == 0:
            print("Divide: Division by zero ⚠️")
            self.rets = self.Returns(quotient=None)
        else:
            q = self.args.numerator // self.args.denominator
            print(f"Divide: {self.args.numerator} // {self.args.denominator} = {q}")
            self.rets = self.Returns(quotient=q)
        return self.rets

class Modulus(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        a: int
        b: int

    class Returns(BaseModel):
        remainder: int

    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self):
        rem = self.args.a % self.args.b
        print(f"Modulus: {self.args.a} % {self.args.b} = {rem}")
        self.rets = self.Returns(remainder=rem)
        return self.rets
    
class Compare(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        a: int
        b: int

    class Returns(BaseModel):
        greater: bool

    args: Arguments | None = None
    rets: Returns | None = None

    def __call__(self):
        result = self.args.a > self.args.b
        print(f"Compare: {self.args.a} > {self.args.b} = {result}")
        self.rets = self.Returns(greater=result)
        return self.rets

class End(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        valid: bool

    args: Arguments | None = None

    def __call__(self):
        status = "Success ✅" if self.args.valid else "Failure ❌"
        print("End:", status)

# -------- Main --------
if __name__ == "__main__":
    engine = MermaidWorkflowEngine(model_registry = {
                'Start':Start,
                'Add':Add,
                'Subtract':Subtract,
                'Multiply':Multiply,
                'ValidateResult':ValidateResult,
                'Divide':Divide,
                'Modulus':Modulus,
                'Compare':Compare,
                'End':End,                
            })
    
    results = engine.run("""
graph TD
    Start["{'para': {'x': 10, 'y': 5, 'z': 3}}"]
    Multiply["{'para': {'factor': 4}}"]

    Start -- "{'x':'a','y':'b'}" --> Add
    Start -- "{'y':'b'}" --> Subtract
    Add -- "{'sum':'a'}" --> Subtract
    Subtract -- "{'result':'x'}" --> Multiply
    Multiply --> ValidateResult
    ValidateResult --> End
""")
    
    results = engine.run("""
graph TD
    Start["{'para': {'x': 20, 'y': 6, 'z': 2}}"]

    Start -- "{'x':'numerator','y':'denominator'}" --> Divide
    Start -- "{'x':'a','z':'b'}" --> Modulus
    Divide -- "{'quotient':'product'}" --> ValidateResult
    ValidateResult --> End
""")
    
    results = engine.run("""
graph TD
    Start["{'para': {'x': 8, 'y': 4, 'z': 0}}"]
    Multiply["{'para': {'factor': 3}}"]

    Start -- "{'x':'a','y':'b'}" --> Compare
    Start --> Multiply
    Start -- "{'x':'numerator','y':'denominator'}" --> Divide
    Multiply --> ValidateResult
    Divide --> ValidateResult
    ValidateResult --> End
""")
    
    results = engine.run("""
graph TD
    Start["{'para': {'x': 7, 'y': 3, 'z': 2}}"]
    Multiply["{'para': {'factor': 3}}"]

    Start -- "{'x':'a','y':'b'}" --> Add
    Add -- "{'sum':'x'}" --> Multiply
    Start -- "{'x':'a','y':'b'}" --> Subtract
    Multiply -- "{'product':'a'}" --> Modulus
    Subtract -- "{'result':'b'}" --> Modulus
    Modulus -- "{'remainder':'product'}" --> ValidateResult
    ValidateResult --> End
""")

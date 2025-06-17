from pydantic import BaseModel, Field
from typing import Optional
from MermaidWorkflowEngine import MermaidWorkflowFunction,MermaidWorkflowEngine
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="MathServer", stateless_http=True)


@mcp.tool(description="Initialize workflow with three integer parameters x, y, z and returns them unchanged. Acts as the starting point for the workflow.")
class Start(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        x: int = Field(..., description="First integer input")
        y: int = Field(..., description="Second integer input")
        z: int = Field(..., description="Third integer input")

    class Returns(BaseModel):
        x: int = Field(..., description="Same as input x")
        y: int = Field(..., description="Same as input y")
        z: int = Field(..., description="Same as input z")

    para: Parameters = Parameters(x=10, y=5, z=3)
    rets: Optional[Returns] = None

    def __init__(self, para: Parameters, rets: Optional[Returns] = None):
        super().__init__(para=para,rets=rets)
        self()
        
    def __call__(self):
        print("Start: Inputs ->", self.para)
        self.rets = self.Returns(**self.para.model_dump())
        return self.rets


@mcp.tool(description="Adds two integers a and b.")
class Add(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        a: int = Field(..., description="First addend")
        b: int = Field(..., description="Second addend")

    class Returns(BaseModel):
        sum: int = Field(..., description="Sum of a and b")

    args: Optional[Arguments] = None
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, rets: Optional[Returns] = None):
        super().__init__(args=args, rets=rets)
        self()
        
    def __call__(self):
        result = self.args.a + self.args.b
        print(f"Add: {self.args.a} + {self.args.b} = {result}")
        self.rets = self.Returns(sum=result)
        return self.rets


@mcp.tool(description="Subtracts b from a.")
class Subtract(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        a: int = Field(..., description="Minuend")
        b: int = Field(..., description="Subtrahend")

    class Returns(BaseModel):
        result: int = Field(..., description="Result of a - b")

    args: Optional[Arguments] = None
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, rets: Optional[Returns] = None):
        super().__init__(args=args, rets=rets)
        self()
        
    def __call__(self):
        result = self.args.a - self.args.b
        print(f"Subtract: {self.args.a} - {self.args.b} = {result}")
        self.rets = self.Returns(result=result)
        return self.rets


@mcp.tool(description="Multiplies x by a constant factor.")
class Multiply(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        factor: int = Field(2, description="Multiplication factor")

    class Arguments(BaseModel):
        x: int = Field(..., description="Value to multiply")

    class Returns(BaseModel):
        product: int = Field(..., description="Product of x and factor")

    para: Parameters = Parameters(factor=4)
    args: Optional[Arguments] = None
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, para: Parameters = Parameters(), rets: Optional[Returns] = None):
        super().__init__(args=args, para=para, rets=rets)
        self()
        
    def __call__(self):
        product = self.args.x * self.para.factor
        print(f"Multiply: {self.args.x} * {self.para.factor} = {product}")
        self.rets = self.Returns(product=product)
        return self.rets


@mcp.tool(description="Validates that the product is a positive number.")
class ValidateResult(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        product: int = Field(..., description="The result to validate")

    class Returns(BaseModel):
        valid: bool = Field(..., description="True if product > 0")

    args: Optional[Arguments] = None
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, rets: Optional[Returns] = None):
        super().__init__(args=args, rets=rets)
        self()
        
    def __call__(self):
        is_valid = self.args.product > 0
        print(f"ValidateResult: {self.args.product} > 0 = {is_valid}")
        self.rets = self.Returns(valid=is_valid)
        return self.rets


@mcp.tool(description="Performs integer division of numerator by denominator, with safe zero-division handling.")
class Divide(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        numerator: int = Field(..., description="Numerator of the division")
        denominator: int = Field(..., description="Denominator of the division")

    class Returns(BaseModel):
        quotient: Optional[int] = Field(None, description="Quotient if valid; None if division by zero")

    args: Optional[Arguments] = None
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, rets: Optional[Returns] = None):
        super().__init__(args=args, rets=rets)
        self()
        
    def __call__(self):
        if self.args.denominator == 0:
            print("Divide: Division by zero ⚠️")
            self.rets = self.Returns(quotient=None)
        else:
            q = self.args.numerator // self.args.denominator
            print(f"Divide: {self.args.numerator} // {self.args.denominator} = {q}")
            self.rets = self.Returns(quotient=q)
        return self.rets


@mcp.tool(description="Calculates the remainder of a divided by b.")
class Modulus(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        a: int = Field(..., description="Dividend")
        b: int = Field(..., description="Divisor")

    class Returns(BaseModel):
        remainder: int = Field(..., description="Remainder of a % b")

    args: Optional[Arguments] = None
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, rets: Optional[Returns] = None):
        super().__init__(args=args, rets=rets)
        self()
        
    def __call__(self):
        rem = self.args.a % self.args.b
        print(f"Modulus: {self.args.a} % {self.args.b} = {rem}")
        self.rets = self.Returns(remainder=rem)
        return self.rets


@mcp.tool(description="Compares two integers and returns True if a > b.")
class Compare(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        a: int = Field(..., description="First value to compare")
        b: int = Field(..., description="Second value to compare")

    class Returns(BaseModel):
        greater: bool = Field(..., description="True if a > b")

    args: Optional[Arguments] = None
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, rets: Optional[Returns] = None):
        super().__init__(args=args, rets=rets)
        self()
        
    def __call__(self):
        result = self.args.a > self.args.b
        print(f"Compare: {self.args.a} > {self.args.b} = {result}")
        self.rets = self.Returns(greater=result)
        return self.rets


@mcp.tool(description="Ends the workflow by reporting success if valid is True.")
class End(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        valid: bool = Field(..., description="Final result status to evaluate")

    args: Optional[Arguments] = None

    def __init__(self, args: Arguments):
        super().__init__(args=args)
        self()
        
    def __call__(self):
        status = "Success ✅" if self.args.valid else "Failure ❌"
        print("End:", status)

# -------- Main --------
if __name__ == "__main__":
    mcp.run(transport="stdio")
#     engine = MermaidWorkflowEngine(model_registry = {
#                 'Start':Start,
#                 'Add':Add,
#                 'Subtract':Subtract,
#                 'Multiply':Multiply,
#                 'ValidateResult':ValidateResult,
#                 'Divide':Divide,
#                 'Modulus':Modulus,
#                 'Compare':Compare,
#                 'End':End,                
#             })
    
#     results = engine.run("""
# graph TD
#     Start["{'para': {'x': 10, 'y': 5, 'z': 3}}"]
#     Multiply["{'para': {'factor': 4}}"]

#     Start -- "{'x':'a','y':'b'}" --> Add
#     Start -- "{'y':'b'}" --> Subtract
#     Add -- "{'sum':'a'}" --> Subtract
#     Subtract -- "{'result':'x'}" --> Multiply
#     Multiply --> ValidateResult
#     ValidateResult --> End
# """,lambda obj,args:obj)
    
#     results = engine.run("""
# graph TD
#     Start["{'para': {'x': 20, 'y': 6, 'z': 2}}"]

#     Start -- "{'x':'numerator','y':'denominator'}" --> Divide
#     Start -- "{'x':'a','z':'b'}" --> Modulus
#     Divide -- "{'quotient':'product'}" --> ValidateResult
#     ValidateResult --> End
# """,lambda obj,args:obj)
    
#     results = engine.run("""
# graph TD
#     Start["{'para': {'x': 8, 'y': 4, 'z': 0}}"]
#     Multiply["{'para': {'factor': 3}}"]

#     Start -- "{'x':'a','y':'b'}" --> Compare
#     Start --> Multiply
#     Start -- "{'x':'numerator','y':'denominator'}" --> Divide
#     Multiply --> ValidateResult
#     Divide --> ValidateResult
#     ValidateResult --> End
# """,lambda obj,args:obj)
    
#     results = engine.run("""
# graph TD
#     Start["{'para': {'x': 7, 'y': 3, 'z': 2}}"]
#     Multiply["{'para': {'factor': 3}}"]

#     Start -- "{'x':'a','y':'b'}" --> Add
#     Add -- "{'sum':'x'}" --> Multiply
#     Start -- "{'x':'a','y':'b'}" --> Subtract
#     Multiply -- "{'product':'a'}" --> Modulus
#     Subtract -- "{'result':'b'}" --> Modulus
#     Modulus -- "{'remainder':'product'}" --> ValidateResult
#     ValidateResult --> End
# """,lambda obj,args:obj)

from pydantic import BaseModel, Field
from typing import Optional
from MermaidWorkflowEngine import MermaidWorkflowFunction, MermaidWorkflowEngine
from mcp.server.fastmcp import FastMCP
from PIL import Image
import os

mcp = FastMCP(name="ImageServer", stateless_http=True)


@mcp.tool(description="Validates if the input image path exists.")
class LoadImage(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        path: str = Field(..., description="Path to the image file")

    class Returns(BaseModel):
        exists: bool = Field(..., description="True if file exists")
        path: str = Field(..., description="Image path")

    para: Parameters
    rets: Optional[Returns] = None

    def __init__(self, para: Parameters, rets: Optional[Returns] = None):
        super().__init__(para=para, rets=rets)
        self()

    def __call__(self):
        try:
            exists = os.path.exists(self.para.path)
            if not exists:
                raise FileNotFoundError(f"File does not exist: {self.para.path}")
            self.rets = self.Returns(exists=exists, path=self.para.path)
            return self.rets
        except Exception as e:
            raise ValueError(f"LoadImage failed: {e}")


@mcp.tool(description="Resize an image to given width and height.")
class ResizeImage(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        width: int
        height: int

    class Arguments(BaseModel):
        path: str

    class Returns(BaseModel):
        path: str

    para: Parameters
    args: Arguments
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, para: Parameters, rets: Optional[Returns] = None):
        super().__init__(args=args, para=para, rets=rets)
        self()

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Input image not found at: {self.args.path}")
            img = Image.open(self.args.path)
            img = img.resize((self.para.width, self.para.height))
            output_path = f"{os.path.splitext(self.args.path)[0]}_resized.jpg"
            img.save(output_path)
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"ResizeImage failed: {e}")


@mcp.tool(description="Rotate an image by a given angle.")
class RotateImage(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        angle: int

    class Arguments(BaseModel):
        path: str

    class Returns(BaseModel):
        path: str

    para: Parameters
    args: Arguments
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, para: Parameters, rets: Optional[Returns] = None):
        super().__init__(args=args, para=para, rets=rets)
        self()

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Input image not found at: {self.args.path}")
            img = Image.open(self.args.path)
            img = img.rotate(self.para.angle)
            output_path = f"{os.path.splitext(self.args.path)[0]}_rotated.jpg"
            img.save(output_path)
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"RotateImage failed: {e}")


@mcp.tool(description="Convert an image to grayscale.")
class GrayscaleImage(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        path: str

    class Returns(BaseModel):
        path: str

    args: Arguments
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, rets: Optional[Returns] = None):
        super().__init__(args=args, rets=rets)
        self()

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Image not found: {self.args.path}")
            img = Image.open(self.args.path).convert("L")
            output_path = f"{os.path.splitext(self.args.path)[0]}_gray.jpg"
            img.save(output_path)
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"GrayscaleImage failed: {e}")


@mcp.tool(description="Crop the image to a specific box (left, upper, right, lower).")
class CropImage(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        left: int
        upper: int
        right: int
        lower: int

    class Arguments(BaseModel):
        path: str

    class Returns(BaseModel):
        path: str

    para: Parameters
    args: Arguments
    rets: Optional[Returns]

    def __init__(self, args: Arguments, para: Parameters, rets: Optional[Returns] = None):
        super().__init__(args=args, para=para, rets=rets)
        self()

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Image not found: {self.args.path}")
            img = Image.open(self.args.path)
            img = img.crop((self.para.left, self.para.upper, self.para.right, self.para.lower))
            output_path = f"{os.path.splitext(self.args.path)[0]}_cropped.jpg"
            img.save(output_path)
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"CropImage failed: {e}")


@mcp.tool(description="Flip an image horizontally or vertically.")
class FlipImage(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        mode: str = Field(..., description="'horizontal' or 'vertical'")

    class Arguments(BaseModel):
        path: str

    class Returns(BaseModel):
        path: str

    para: Parameters
    args: Arguments
    rets: Optional[Returns]

    def __init__(self, args: Arguments, para: Parameters, rets: Optional[Returns] = None):
        super().__init__(args=args, para=para, rets=rets)
        self()

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Image not found: {self.args.path}")
            img = Image.open(self.args.path)
            if self.para.mode == "horizontal":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif self.para.mode == "vertical":
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                raise ValueError("Invalid mode for FlipImage. Use 'horizontal' or 'vertical'.")
            output_path = f"{os.path.splitext(self.args.path)[0]}_flipped.jpg"
            img.save(output_path)
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"FlipImage failed: {e}")


@mcp.tool(description="Ends the workflow and reports image output path.")
class EndImage(MermaidWorkflowFunction):
    class Arguments(BaseModel):
        path: str = Field(..., description="Final image path")

    args: Arguments

    def __init__(self, args: Arguments):
        super().__init__(args=args)
        self()

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Final image not found: {self.args.path}")
            print(f"EndImage: Final image saved at {self.args.path}")
        except Exception as e:
            raise ValueError(f"EndImage failed: {e}")


# -------- Main --------
if __name__ == "__main__":
    mcp.run(transport="stdio")

#     engine = MermaidWorkflowEngine(model_registry = {
#         'LoadImage':LoadImage,
#         'ResizeImage':ResizeImage,
#         'RotateImage':RotateImage,
#         'GrayscaleImage':GrayscaleImage,
#         'CropImage':CropImage,
#         'FlipImage':FlipImage,
#         'EndImage':EndImage,
#             })
    
#     results = engine.run("""
# graph TD
#     LoadImage["{'para': {'path': 'input.jpg'}}"]

#     ResizeImage["{'para': {'width': 256, 'height': 256}}"]
#     GrayscaleImage["{}"]
#     FlipImage["{'para': {'mode': 'horizontal'}}"]
#     RotateImage["{'para': {'angle': 90}}"]
#     EndImage["{}"]

#     LoadImage -- "{'path':'path'}" --> ResizeImage
#     ResizeImage -- "{'path':'path'}" --> GrayscaleImage
#     GrayscaleImage -- "{'path':'path'}" --> FlipImage
#     FlipImage -- "{'path':'path'}" --> RotateImage
#     RotateImage -- "{'path':'path'}" --> EndImage
# """)

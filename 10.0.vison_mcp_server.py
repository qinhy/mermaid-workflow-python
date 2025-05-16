from pydantic import BaseModel, Field
from typing import Optional
from MermaidWorkflowEngine import MermaidWorkflowFunction, MermaidWorkflowEngine
from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
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


@mcp.tool(description="Apply a blur filter to an image.")
class BlurImage(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        radius: float = Field(..., description="Blur radius, higher values create more blur")

    class Arguments(BaseModel):
        path: str

    class Returns(BaseModel):
        path: str

    para: Parameters
    args: Arguments
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, para: Parameters = None, rets: Optional[Returns] = None):
        super().__init__(args=args, para=para, rets=rets)
        self()

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Image not found: {self.args.path}")
            img = Image.open(self.args.path)
            img = img.filter(ImageFilter.GaussianBlur(radius=self.para.radius))
            output_path = f"{os.path.splitext(self.args.path)[0]}_blurred.jpg"
            img.save(output_path)
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"BlurImage failed: {e}")


@mcp.tool(description="Adjust the brightness, contrast, or saturation of an image.")
class AdjustImage(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        brightness: float = Field(..., description="Brightness factor (0.0-2.0, 1.0 is original)")
        contrast: float = Field(..., description="Contrast factor (0.0-2.0, 1.0 is original)")
        saturation: float = Field(..., description="Saturation factor (0.0-2.0, 1.0 is original)")

    class Arguments(BaseModel):
        path: str

    class Returns(BaseModel):
        path: str

    para: Parameters
    args: Arguments
    rets: Optional[Returns] = None

    def __init__(self, args: Arguments, para: Parameters = None, rets: Optional[Returns] = None):
        if para is None:
            para = self.Parameters()
        super().__init__(args=args, para=para, rets=rets)
        self()

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Image not found: {self.args.path}")
            img = Image.open(self.args.path)
            
            # Apply brightness adjustment
            if self.para.brightness != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(self.para.brightness)
            
            # Apply contrast adjustment
            if self.para.contrast != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(self.para.contrast)
                
            # Apply saturation adjustment
            if self.para.saturation != 1.0:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(self.para.saturation)
                
            output_path = f"{os.path.splitext(self.args.path)[0]}_adjusted.jpg"
            img.save(output_path)
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"AdjustImage failed: {e}")


@mcp.tool(description="Apply various filters to an image.")
class FilterImage(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        filter_type: str = Field(..., description="Filter type: 'emboss', 'find_edges', 'contour', 'sharpen', or 'smooth'")

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
                raise FileNotFoundError(f"Image not found: {self.args.path}")
            img = Image.open(self.args.path)
            
            filter_map = {
                'emboss': ImageFilter.EMBOSS,
                'find_edges': ImageFilter.FIND_EDGES,
                'contour': ImageFilter.CONTOUR,
                'sharpen': ImageFilter.SHARPEN,
                'smooth': ImageFilter.SMOOTH
            }
            
            if self.para.filter_type not in filter_map:
                raise ValueError(f"Invalid filter type: {self.para.filter_type}. Valid options are: {', '.join(filter_map.keys())}")
                
            img = img.filter(filter_map[self.para.filter_type])
            output_path = f"{os.path.splitext(self.args.path)[0]}_{self.para.filter_type}.jpg"
            img.save(output_path)
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"FilterImage failed: {e}")


@mcp.tool(description="Add a watermark text to an image.")
class WatermarkImage(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        text: str = Field(..., description="Text to add as watermark")
        position: str = Field(..., description="Position: 'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'")
        opacity: float = Field(..., description="Opacity of watermark (0.0-1.0)")

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
                raise FileNotFoundError(f"Image not found: {self.args.path}")
                
            from PIL import ImageDraw, ImageFont
            
            # Open the original image
            img = Image.open(self.args.path).convert("RGBA")
            
            # Create a transparent overlay for the watermark
            overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Try to use a default font, or fall back to default
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()
                
            bbox = draw.textbbox((0, 0), self.para.text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Determine position
            position_map = {
                'center': ((img.width - text_width) // 2, (img.height - text_height) // 2),
                'top_left': (10, 10),
                'top_right': (img.width - text_width - 10, 10),
                'bottom_left': (10, img.height - text_height - 10),
                'bottom_right': (img.width - text_width - 10, img.height - text_height - 10)
            }
            
            if self.para.position not in position_map:
                position = position_map['center']
            else:
                position = position_map[self.para.position]
            
            # Draw the text with transparency
            draw.text(position, self.para.text, font=font, fill=(0, 0, 0, int(255 * self.para.opacity)))
            
            # Composite the watermark with the original image
            watermarked = Image.alpha_composite(img, overlay)
            watermarked = watermarked.convert("RGB")  # Convert back to RGB for saving as JPG
            
            output_path = f"{os.path.splitext(self.args.path)[0]}_watermarked.jpg"
            watermarked.save(output_path)
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"WatermarkImage failed: {e}")


@mcp.tool(description="Convert an image to a different format.")
class ConvertImageFormat(MermaidWorkflowFunction):
    class Parameters(BaseModel):
        format: str = Field(..., description="Target format: 'jpg', 'png', 'bmp', 'gif', 'webp'")
        quality: int = Field(..., description="Quality for lossy formats (1-100)")

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
                raise FileNotFoundError(f"Image not found: {self.args.path}")
                
            valid_formats = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp']
            if self.para.format.lower() not in valid_formats:
                raise ValueError(f"Invalid format: {self.para.format}. Valid formats are: {', '.join(valid_formats)}")
                
            img = Image.open(self.args.path)
            
            # Normalize format name
            fmt = self.para.format.lower()
            if fmt == 'jpg':
                fmt = 'jpeg'
                
            # Create output path with new extension
            output_path = f"{os.path.splitext(self.args.path)[0]}.{self.para.format.lower()}"
            
            # Save with quality parameter for formats that support it
            if fmt in ['jpeg', 'webp']:
                img.save(output_path, format=fmt.upper(), quality=self.para.quality)
            else:
                img.save(output_path, format=fmt.upper())
                
            self.rets = self.Returns(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"ConvertImageFormat failed: {e}")


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
#         'BlurImage':BlurImage,
#         'AdjustImage':AdjustImage,
#         'FilterImage':FilterImage,
#         'WatermarkImage':WatermarkImage,
#         'ConvertImageFormat':ConvertImageFormat,
#         'EndImage':EndImage,
#             })
    
#     results = engine.run("""
# graph TD
#     LoadImage["{'para': {'path': './input.jpg'}}"]
#     ResizeImage["{'para': {'width': 512, 'height': 512}}"]
#     CropImage["{'para': {'left': 50, 'upper': 50, 'right': 462, 'lower': 462}}"]
#     BlurImage["{'para': {'radius': 3}}"]
#     RotateImage["{'para': {'angle': 270}}"]
#     AdjustImage["{'para': {'brightness': 1.2, 'contrast': 1.3, 'saturation': 0.9}}"]
#     FlipImage["{'para': {'mode': 'vertical'}}"]
#     WatermarkImage["{'para': {'text':'CONFIDENTIAL', 'position':'bottom_right', 'opacity':0.5}}"]
#     FilterImage["{'para': {'filter_type': 'sharpen'}}"]
#     ConvertImageFormat["{'para': {'format': 'png', 'quality':90}}"]

#     LoadImage -- "{'path':'path'}" --> ResizeImage
#     ResizeImage -- "{'path':'path'}" --> CropImage
#     CropImage -- "{'path':'path'}" --> BlurImage
#     BlurImage -- "{'path':'path'}" --> GrayscaleImage
#     GrayscaleImage -- "{'path':'path'}" --> RotateImage

#     RotateImage -- "{'path':'path'}" --> AdjustImage
#     AdjustImage -- "{'path':'path'}" --> FlipImage
#     FlipImage -- "{'path':'path'}" --> WatermarkImage
#     WatermarkImage -- "{'path':'path'}" --> FilterImage
#     FilterImage -- "{'path':'path'}" --> ConvertImageFormat
#     ConvertImageFormat -- "{'path':'path'}" --> EndImage
# """)

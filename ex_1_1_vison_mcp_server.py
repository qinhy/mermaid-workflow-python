from io import BytesIO
import tempfile
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Type

import requests
from MermaidWorkflowEngine import MermaidWorkflowFunction, MermaidWorkflowEngine
from mcp.server.fastmcp import FastMCP
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import os

from RSAjson import load_RSA

mcp = FastMCP(name="ImageServer", stateless_http=True)


class LoadImage(MermaidWorkflowFunction):
    description:str = Field("Validates if the input image path exists.")

    class Parameter(BaseModel):
        path: str = Field(..., description="Path to the image file")

    class Returness(BaseModel):
        exists: bool = Field(..., description="True if file exists")
        path: str = Field(..., description="Image path")

    para: Parameter
    rets: Optional[Returness] = None

    def __call__(self):
        try:
            exists = os.path.exists(self.para.path)
            if not exists:
                raise FileNotFoundError(f"File does not exist: {self.para.path}")
            self.rets = self.Returness(exists=exists, path=self.para.path)
            return self.rets
        except Exception as e:
            raise ValueError(f"LoadImage failed: {e}")


class ResizeImage(MermaidWorkflowFunction):
    description:str = Field("Resize an image to given width and height.")

    class Parameter(BaseModel):
        width: int
        height: int

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Input image not found at: {self.args.path}")
            img = Image.open(self.args.path)
            img = img.resize((self.para.width, self.para.height))
            output_path = f"{os.path.splitext(self.args.path)[0]}_resized.jpg"
            img.save(output_path)
            self.rets = self.Returness(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"ResizeImage failed: {e}")


class RotateImage(MermaidWorkflowFunction):
    description:str = Field("Rotate an image by a given angle.")

    class Parameter(BaseModel):
        angle: int

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Input image not found at: {self.args.path}")
            img = Image.open(self.args.path)
            img = img.rotate(self.para.angle)
            output_path = f"{os.path.splitext(self.args.path)[0]}_rotated.jpg"
            img.save(output_path)
            self.rets = self.Returness(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"RotateImage failed: {e}")

class GrayscaleImage(MermaidWorkflowFunction):
    description:str = Field("Convert an image to 8-bit grayscale (ImageJ style).")

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    args: Arguments
    rets: Optional[Returness] = None

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Image not found: {self.args.path}")

            img = Image.open(self.args.path)
            img_mode = img.mode

            # Convert to numpy array for detailed control
            img_np = np.array(img)            

            # If it's not 8-bit, rescale to 0–255
            if img_np.dtype == np.uint16:
                img_np = (img_np / 256).astype(np.uint8)
            elif img_np.dtype == np.float32 or img_np.dtype == np.float64:
                img_np = np.clip(img_np, 0, 1)  # normalize if needed
                img_np = (img_np * 255).astype(np.uint8)

            # Convert RGB to grayscale if needed
            if img_mode in ["RGB", "RGBA"]:
                # Use weighted RGB to grayscale conversion (ITU-R BT.601)
                # img_np = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])
                img_np = img_np.mean(2).astype(np.uint8)

            # Convert back to PIL and save
            img_8bit = Image.fromarray(img_np, mode='L')
            output_path = f"{os.path.splitext(self.args.path)[0]}_gray.jpg"
            img_8bit.save(output_path)

            self.rets = self.Returness(path=output_path)
            return self.rets

        except Exception as e:
            raise ValueError(f"GrayscaleImage failed: {e}")


class CropImage(MermaidWorkflowFunction):
    description:str = Field("Crop the image to a specific box (left, upper, right, lower).")

    class Parameter(BaseModel):
        left: int
        upper: int
        right: int
        lower: int

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None
 
    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Image not found: {self.args.path}")
            img = Image.open(self.args.path)
            img = img.crop((self.para.left, self.para.upper, self.para.right, self.para.lower))
            output_path = f"{os.path.splitext(self.args.path)[0]}_cropped.jpg"
            img.save(output_path)
            self.rets = self.Returness(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"CropImage failed: {e}")


class FlipImage(MermaidWorkflowFunction):
    description:str = Field("Flip an image horizontally or vertically.")

    class Parameter(BaseModel):
        mode: str = Field(..., description="'horizontal' or 'vertical'")

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None

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
            self.rets = self.Returness(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"FlipImage failed: {e}")


class EndImage(MermaidWorkflowFunction):
    description:str = Field("Ends the workflow and reports image output path.")

    class Arguments(BaseModel):
        path: str = Field(..., description="Final image path")

    args: Arguments

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Final image not found: {self.args.path}")
            print(f"EndImage: Final image saved at {self.args.path}")
        except Exception as e:
            raise ValueError(f"EndImage failed: {e}")


class BlurImage(MermaidWorkflowFunction):
    description:str = Field("Apply a blur filter to an image.")

    class Parameter(BaseModel):
        radius: float = Field(..., description="Blur radius, higher values create more blur")

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None

    def __call__(self):
        try:
            if not os.path.exists(self.args.path):
                raise FileNotFoundError(f"Image not found: {self.args.path}")
            img = Image.open(self.args.path)
            img = img.filter(ImageFilter.GaussianBlur(radius=self.para.radius))
            output_path = f"{os.path.splitext(self.args.path)[0]}_blurred.jpg"
            img.save(output_path)
            self.rets = self.Returness(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"BlurImage failed: {e}")


class AdjustImage(MermaidWorkflowFunction):
    description:str = Field("Adjust the brightness, contrast, or saturation of an image.")

    class Parameter(BaseModel):
        brightness: float = Field(..., description="Brightness factor (0.0-2.0, 1.0 is original)")
        contrast: float = Field(..., description="Contrast factor (0.0-2.0, 1.0 is original)")
        saturation: float = Field(..., description="Saturation factor (0.0-2.0, 1.0 is original)")

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None

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
            self.rets = self.Returness(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"AdjustImage failed: {e}")


class FilterImage(MermaidWorkflowFunction):
    description:str = Field("Apply various filters to an image.")

    class Parameter(BaseModel):
        filter_type: str = Field(..., description="Filter type: 'emboss', 'find_edges', 'contour', 'sharpen', or 'smooth'")

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None

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
            self.rets = self.Returness(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"FilterImage failed: {e}")


class WatermarkImage(MermaidWorkflowFunction):
    description:str = Field("Add a watermark text to an image.")

    class Parameter(BaseModel):
        text: str = Field(..., description="Text to add as watermark")
        position: str = Field(..., description="Position: 'center', 'top_left', 'top_right', 'bottom_left', 'bottom_right'")
        opacity: float = Field(..., description="Opacity of watermark (0.0-1.0)")

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None

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
            self.rets = self.Returness(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"WatermarkImage failed: {e}")


class ConvertImageFormat(MermaidWorkflowFunction):
    description:str = Field("Convert an image to a different format.")

    class Parameter(BaseModel):
        format: str = Field(..., description="Target format: 'jpg', 'png', 'bmp', 'gif', 'webp'")
        quality: int = Field(..., description="Quality for lossy formats (1-100)")

    class Arguments(BaseModel):
        path: str

    class Returness(BaseModel):
        path: str

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None

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
                
            self.rets = self.Returness(path=output_path)
            return self.rets
        except Exception as e:
            raise ValueError(f"ConvertImageFormat failed: {e}")
        
class ImageTiler(MermaidWorkflowFunction):
    description:str = Field("Tiles images into a grid.")

    class Parameter(BaseModel):
        img_size_limit: int = Field(250000000, description="Maximum image size limit in pixels")
        cols: int = Field(2, description="Number of columns for tiling images")
        rows: int = Field(2, description="Number of rows for tiling images")
        final_width: int = Field(800, description="Final image width in pixels")
        final_height: int = Field(600, description="Final image height in pixels")
        output_format: str = Field("jpg", description="Output image format (jpg, png, etc.)")
        output_path: Optional[str] = Field('tiled_image.jpg', description="Custom output path for the tiled image")
        
    class Arguments(BaseModel):
        image_sources: List[str] = Field(
            [],
            description="List of image URLs or local file paths, depending on the mode parameter."
        )
        image_slices: List[Tuple[float, float]] = Field(
            [],
            description="List of normalized coordinates (x,y) in range -1.0 to 1.0 that to do slicing to sub each image , with: width * x, height * y"
        )
        
    class Returness(BaseModel):
        path: Optional[str] = Field(
            None,
            description="Path to the saved tiled image"
        )

    para: Parameter
    args: Arguments
    rets: Optional[Returness] = None

    def __call__(self, *args, **kwargs):
        print("Starting image tiling process.")            
        
        if not self._validate_inputs():
            return self
            
        images = self._process_images()
        
        if not images:
            print("No valid images were processed. Exiting.")
            self.rets = self.Returness(path="")
            return self                
        
        self._create_tiled_image(images)
        
        return self

    def _validate_inputs(self):
        """Validate input parameters and sources."""
        sources = self.args.image_sources
        if not sources:
            print("No image sources provided. Please provide at least one image source.")
            return False
            
        if self.para.cols <= 0 or self.para.rows <= 0:
            print("Columns and rows must be positive integers.")
            return False
            
        if self.para.final_width <= 0 or self.para.final_height <= 0:
            print("Final width and height must be positive integers.")
            return False
            
        return True
        
    def _process_images(self):
        """Process all image sources and return a list of processed images."""
        sources = self.args.image_sources
        cols = self.para.cols
        rows = self.para.rows
        final_width = self.para.final_width
        final_height = self.para.final_height
        
        # Calculate dimensions for each cell in the grid
        cell_width = final_width // cols
        cell_height = final_height // rows
        print(
            f"Tiling parameters: {cols} cols x {rows} rows; Final size: {final_width}x{final_height}; Each cell: {cell_width}x{cell_height}."
        )
        
        # List to store processed images
        images = []
        for idx, source in enumerate(sources):            
            try:
                img = self._load_image(source, idx, len(sources))
                if img is None:
                    continue
                
                # Apply image slicing if specified
                img = self._apply_slicing(img, idx)
                
                # Resize image to fit cell dimensions
                original_size = img.size
                img = img.resize((cell_width, cell_height), Image.LANCZOS)
                print(
                    f"Resized image {idx+1} from {original_size} to {img.size}."
                )
                
                images.append(img)
            except Exception as e:
                print(f"Unexpected error processing image at {source}: {str(e)}")
                continue
                
        return images
        
    def _load_image(self, source, idx, total_sources):
        """Load an image from a URL or local file path."""
        try:
            # Set PIL's maximum image size limit to prevent decompression bomb attacks
            Image.MAX_IMAGE_PIXELS = self.para.img_size_limit

            if source.startswith(("http://", "https://")):
                print(
                    f"Downloading image {idx + 1}/{total_sources} from {source}."
                )
                response = requests.get(source, timeout=10)
                response.raise_for_status()
                img_data = BytesIO(response.content)
                
                img = Image.open(img_data)
                print(
                    f"Successfully downloaded image {idx+1}: {img.format}, size: {img.size}, mode: {img.mode}"
                )
            else:
                if not os.path.exists(source):
                    raise FileNotFoundError(f"Image file not found: {source}")
                print(
                    f"Opening local image {idx + 1}/{total_sources} from {source}."
                )
                    
                img = Image.open(source)
                print(
                    f"Successfully opened image {idx+1}: {img.format}, size: {img.size}, mode: {img.mode}"
                )
            
            # Convert image to RGB if necessary
            if img.mode != 'RGB':
                print(
                    f"Converting image {idx+1} from {img.mode} to RGB mode."
                )
                img = img.convert('RGB')
                
            return img
        except requests.exceptions.RequestException as e:
            print(f"Network error while downloading image {idx+1} from {source}: {str(e)}")
        except FileNotFoundError as e:
            print(f"{str(e)}")
        except Image.UnidentifiedImageError:
            print(f"Could not identify image {idx+1} format for {source}")
        except Image.DecompressionBombError as e:
            print(f"Image {idx+1} from {source} is too large: {str(e)}")
        
        return None
        
    def _apply_slicing(self, img, idx):
        """Apply slicing to an image if specified in the model args."""
        slices = self.args.image_slices
        if not slices or idx >= len(slices) or not slices[idx] or len(slices[idx])!=2:
            return img
            
        rx, ry = slices[idx]
        width, height = img.size
        
        # Calculate target dimensions based on normalized coordinates
        target_width = int(abs(rx) * width)
        target_height = int(abs(ry) * height)
        
        # Calculate starting positions
        start_x = 0 if rx > 0 else width - target_width
        start_y = 0 if ry > 0 else height - target_height
        
        # Crop the image
        cropped_img = img.crop((start_x, start_y, start_x + target_width, start_y + target_height))
        print(
            f"Applied slicing to image {idx+1}: coordinates ({rx}, {ry}), resulting size: {cropped_img.size}"
        )
        
        return cropped_img
        
    def _create_tiled_image(self, images):
        """Create and save the final tiled image."""
        cols = self.para.cols
        rows = self.para.rows
        final_width = self.para.final_width
        final_height = self.para.final_height
        cell_width = final_width // cols
        cell_height = final_height // rows
        
        # Create a blank canvas for the final tiled image
        tiled_image = Image.new('RGB', (final_width, final_height))
        img_index = 0
        
        print(f"Tiling {len(images)} images into a {rows}x{cols} grid.")

        for row in range(rows):
            for col in range(cols):
                if img_index >= len(images):
                    break  # No more images to paste
                x = col * cell_width
                y = row * cell_height
                tiled_image.paste(images[img_index], (x, y))
                img_index += 1
                print(
                    f"Placed image {img_index} at grid position ({row}, {col}), at array position ({y}, {x})"
                )

        self._save_tiled_image(tiled_image)
        
    def _save_tiled_image(self, tiled_image):
        """Save the tiled image to disk."""
        try:
            # Determine output path
            if self.para.output_path:
                output_path = self.para.output_path
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            else:
                output_format = self.para.output_format.lower()
                if not output_format.startswith('.'):
                    output_format = f".{output_format}"
                
                # Create a temporary file with the specified extension
                with tempfile.NamedTemporaryFile(suffix=output_format, delete=False) as tmp:
                    output_path = tmp.name
            
            # Save the image
            tiled_image.save(output_path)
            print(
                f"Tiled image saved at {output_path}. Final size: {tiled_image.size}"
            )
            self.rets = self.Returness(path=output_path)
        except Exception as e:
            print(f"Error saving tiled image: {str(e)}")
            self.rets = self.Returness(path="")

# -------- Main --------
if __name__ == "__main__":
    def add_tool(t:Type[MermaidWorkflowFunction]):        
        mcp.add_tool(t, 
        description=t.model_fields['description'].default)

    add_tool(LoadImage)
    add_tool(ResizeImage)
    add_tool(RotateImage)
    add_tool(GrayscaleImage)
    add_tool(CropImage)
    add_tool(FlipImage)
    add_tool(EndImage)
    add_tool(BlurImage)
    add_tool(AdjustImage)
    add_tool(FilterImage)
    add_tool(WatermarkImage)
    add_tool(ConvertImageFormat)
    add_tool(ImageTiler)

    mcp.run(transport="stdio")

    # engine = MermaidWorkflowEngine(model_registry = {
    #     'LoadImage':LoadImage,
    #     'ResizeImage':ResizeImage,
    #     'RotateImage':RotateImage,
    #     'GrayscaleImage':GrayscaleImage,
    #     'CropImage':CropImage,
    #     'FlipImage':FlipImage,
    #     'BlurImage':BlurImage,
    #     'AdjustImage':AdjustImage,
    #     'FilterImage':FilterImage,
    #     'WatermarkImage':WatermarkImage,
    #     'ConvertImageFormat':ConvertImageFormat,
    #     'ImageTiler':ImageTiler,
    #     'EndImage':EndImage,
    #         })
    
#     itc = load_RSA("./tmp/image_tiler.json","./tmp/private_key.pem")    
#     aic = {'para':{'brightness':4.5,'contrast':4.5,'saturation':1.0}}
#     results = engine.run(f"""
# graph TD
#     ImageTiler["{itc}"]
#     AdjustImage["{aic}"]

#     ImageTiler --> AdjustImage
#     AdjustImage --> GrayscaleImage
#     GrayscaleImage --> EndImage
# """,lambda obj,args:obj)
    
#     results = engine.run("""
# graph TD
#     LoadImage["{'para': {'path': './tmp/input.jpg'}}"]
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

#     results = engine.run("""
# graph TD
#     LoadImage["{'para': {'path': './tmp/input.jpg'}}"]
#     BlurImage["{'para': {'radius': 2}}"]
#     BlurImage_2["{'para': {'radius': 5}}"]
#     ResizeImage_01["{'para': {'width': 512, 'height': 512}}"]
#     ResizeImage["{'para': {'width': 1024, 'height': 1024}}"]

#     LoadImage -- "{'path':'path'}" --> ResizeImage_01
#     ResizeImage_01 --> BlurImage
#     BlurImage --> BlurImage_2
#     BlurImage_2 --> ResizeImage
#     ResizeImage --> EndImage
# """)

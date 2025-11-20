from PIL import Image

def apply_torch_patches():
    """
    Không còn cần patch thủ công. Hàm này được giữ lại để tương thích (no-op).
    """
    return

def optimize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Resize ảnh nếu quá lớn để tiết kiệm RAM khi xử lý.
    """
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


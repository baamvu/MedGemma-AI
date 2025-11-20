import torch
import gc
import os
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

def load_model_pipeline(model_config):
    """
    Hàm load model thông minh:
    - Tự động detect GPU/CPU
    - Áp dụng quantization phù hợp
    - Load base + adapter hoặc merged model
    """
    print(f"[LOADER] Cấu hình: {model_config}")
    
    base_path = model_config['base']
    adapter_path = model_config['adapter']
    is_local = model_config['is_local']
    
    # 1. Dọn dẹp bộ nhớ trước khi load
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 2. Xác định thiết bị và Quantization Config
    device_map = None
    bnb_config = None
    torch_dtype = torch.float32
    
    if torch.cuda.is_available():
        print("[LOADER] Phát hiện GPU - Kích hoạt chế độ GPU.")
        device_map = "auto"
        torch_dtype = torch.float16
        # Nếu có bitsandbytes, dùng 4-bit để tiết kiệm VRAM
        if BitsAndBytesConfig:
            print("[LOADER] Kích hoạt 4-bit Quantization (GPU).")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
    else:
        print("[LOADER] Không có GPU - Kích hoạt chế độ CPU tiết kiệm RAM.")
        device_map = None # CPU tự quản lý
        torch_dtype = torch.float32

    # 3. Load Processor
    print("[LOADER] Đang load Processor...")
    processor = AutoProcessor.from_pretrained(
        base_path, 
        trust_remote_code=True,
        local_files_only=is_local
    )

    # 4. Load Model
    print(f"[LOADER] Đang load Model từ: {base_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True, # Quan trọng cho máy RAM yếu
        trust_remote_code=True,
        local_files_only=is_local
    )

    # 5. BỎ QUA bước nén INT8 động (gây crash trên máy yếu)
    # Thay vào đó, ta chấp nhận dùng model gốc (float32) nhờ RAM ảo.
    if not torch.cuda.is_available():
        print("[LOADER] Đã load model CPU. Bỏ qua nén INT8 để tránh crash.")
        # Phần nén dưới đây đã bị vô hiệu hóa:
        # try:
        #     model = torch.quantization.quantize_dynamic(
        #         model,
        #         {torch.nn.Linear},
        #         dtype=torch.qint8
        #     )
        #     print("[LOADER] Nén INT8 thành công!")
        # except Exception as e:
        #     print(f"[LOADER] Lỗi nén INT8 (có thể bỏ qua): {e}")

    # 6. Load Adapter (nếu có và không phải merged model)
    if adapter_path:
        print(f"[LOADER] Đang gắn Adapter từ: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("[LOADER] Đã gắn Adapter.")

    # 7. Tối ưu hóa cuối cùng
    model.eval()
    # Tắt gradient để tiết kiệm RAM
    for param in model.parameters():
        param.requires_grad = False

    return processor, model


import streamlit as st
import time
import torch
import gc
from PIL import Image

# Import local modules
from src.utils import apply_torch_patches, optimize_image
from src.config import get_model_paths
from src.model_loader import load_model_pipeline

# 1. Apply patches ngay lập tức
apply_torch_patches()

# 2. Cấu hình trang
st.set_page_config(page_title="MedGemma AI", page_icon="🏥", layout="wide")

# 3. Hàm load model (được cache để không load lại)
@st.cache_resource(show_spinner=False)
def get_ai_system():
    """
    Hàm này chỉ chạy 1 lần duy nhất khi khởi động app.
    """
    paths = get_model_paths()
    return load_model_pipeline(paths)

# --- Sidebar ---
with st.sidebar:
    st.title("🏥 MedGemma AI")
    st.caption("Hệ thống hỗ trợ chẩn đoán hình ảnh y tế.")
    
    # Thông tin hệ thống
    if torch.cuda.is_available():
        st.success("🚀 Đang chạy trên GPU")
    else:
        st.info("💻 Đang chạy trên CPU (INT8)")
        
    st.divider()
    st.markdown("**Hướng dẫn:**")
    st.markdown("1. Upload ảnh X-quang/MRI/CT.")
    st.markdown("2. Nhập câu hỏi hoặc yêu cầu.")
    st.markdown("3. Nhấn **Phân tích**.")

# --- Main UI ---
st.header("Hồ sơ bệnh án điện tử (AI Hỗ trợ)")

# Chỉ load model khi người dùng sẵn sàng (Lazy Load)
if "model_loaded" not in st.session_state:
    st.session_state["model_loaded"] = False

if not st.session_state["model_loaded"]:
    st.info("⚠️ Hệ thống đang ở chế độ chờ để tiết kiệm tài nguyên.")
    if st.button("Khởi động Hệ thống AI", type="primary"):
        with st.spinner("Đang khởi động... Vui lòng đợi (có thể mất 1-2 phút)..."):
            try:
                processor, model = get_ai_system()
                st.session_state["model_loaded"] = True
                st.success("✅ Hệ thống đã sẵn sàng!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Lỗi khởi động model: {e}")
                st.stop()
else:
    # Khi đã load xong thì lấy model từ cache
    processor, model = get_ai_system()

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Tải lên ảnh y tế", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            # Resize ảnh để tiết kiệm RAM
            image = optimize_image(image)
            st.image(image, caption="Ảnh bệnh nhân", use_container_width=True)

    with col2:
        user_prompt = st.text_area(
            "Yêu cầu phân tích:", 
            value="Hãy mô tả chi tiết các bất thường trong ảnh và đưa ra chẩn đoán sơ bộ.",
            height=150
        )
        
        if st.button("Phân tích / Tạo bệnh án", type="primary"):
            if not uploaded_file:
                st.warning("Vui lòng tải ảnh lên trước.")
            else:
                try:
                    with st.spinner("AI đang phân tích..."):
                        # Chuẩn bị input
                        chat = [
                            {
                                "role": "user", 
                                "content": [
                                    {"type": "image"}, 
                                    {"type": "text", "text": user_prompt}
                                ]
                            }
                        ]
                        prompt = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
                        inputs = processor(text=prompt, images=image, return_tensors="pt")
                        
                        # Chuyển sang device phù hợp
                        device = next(model.parameters()).device
                        inputs = {k: v.to(device) for k, v in inputs.items()}

                        # Generate
                        with torch.inference_mode():
                            out_ids = model.generate(
                                **inputs, 
                                max_new_tokens=512,
                                do_sample=True,
                                temperature=0.7,
                                top_k=50
                            )
                        
                        # Decode kết quả
                        result = processor.decode(out_ids[0], skip_special_tokens=True)
                        
                        # Dọn dẹp RAM
                        del inputs
                        del out_ids
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    st.success("Đã phân tích xong!")
                    st.markdown("### Kết quả:")
                    st.write(result)
                    
                except Exception as e:
                    st.error(f"Lỗi khi phân tích: {e}")



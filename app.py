import streamlit as st
import time
import torch
import gc
from PIL import Image
from io import BytesIO

# Import local modules
from src.utils import apply_torch_patches, optimize_image, clean_model_output
from src.config import get_model_paths
from src.model_loader import load_model_pipeline
from src.cache_manager import (
    get_cached_result, save_result, get_file_content_from_upload,
    get_cache_key, clear_cache, get_cache_stats
)

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
    st.markdown("1. Upload một hoặc nhiều ảnh X-quang/MRI/CT.")
    st.markdown("2. Chọn ảnh cần phân tích (nếu có nhiều ảnh).")
    st.markdown("3. Nhập câu hỏi hoặc yêu cầu.")
    st.markdown("4. Nhấn **Phân tích**.")
    
    st.divider()
    
    # Thống kê cache
    cache_stats = get_cache_stats()
    st.caption(f"💾 Cache: {cache_stats['total_entries']} kết quả đã lưu")
    
    # Nút xóa cache (tùy chọn)
    if st.button("🗑️ Xóa cache", help="Xóa toàn bộ cache kết quả phân tích"):
        clear_cache()
        st.success("✅ Đã xóa cache!")
        st.rerun()

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

    # Upload nhiều file
    uploaded_files = st.file_uploader(
        "Tải lên ảnh y tế (có thể chọn nhiều file)", 
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True
    )
    
    # Khởi tạo cache nếu chưa có
    if "cached_images" not in st.session_state:
        st.session_state["cached_images"] = {}
    
    # Khởi tạo cache file content để dùng cho cache kết quả
    if "file_contents" not in st.session_state:
        st.session_state["file_contents"] = {}
    
    # Xử lý và cache tất cả ảnh đã upload
    if uploaded_files:
        with st.spinner(f"Đang xử lý {len(uploaded_files)} ảnh..."):
            for uploaded_file in uploaded_files:
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                
                # Chỉ xử lý nếu chưa có trong cache
                if file_key not in st.session_state["cached_images"]:
                    try:
                        # Lưu file content để dùng cho cache kết quả
                        file_content = get_file_content_from_upload(uploaded_file)
                        st.session_state["file_contents"][file_key] = file_content
                        
                        # Xử lý ảnh
                        image = Image.open(BytesIO(file_content)).convert("RGB")
                        image = optimize_image(image, max_size=512)
                        st.session_state["cached_images"][file_key] = {
                            "image": image,
                            "name": uploaded_file.name
                        }
                    except Exception as e:
                        st.warning(f"⚠️ Không thể đọc file {uploaded_file.name}: {e}")
        
        # Xóa cache của các file không còn trong danh sách
        current_keys = {f"{f.name}_{f.size}" for f in uploaded_files}
        keys_to_remove = [k for k in st.session_state["cached_images"].keys() if k not in current_keys]
        for key in keys_to_remove:
            del st.session_state["cached_images"][key]
            if key in st.session_state.get("file_contents", {}):
                del st.session_state["file_contents"][key]
        
        st.success(f"✅ Đã tải {len(uploaded_files)} ảnh thành công!")
        
        # Hiển thị gallery ảnh
        st.markdown("### 📸 Ảnh đã tải:")
        num_cols = min(3, len(uploaded_files))
        cols = st.columns(num_cols)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_key in st.session_state["cached_images"]:
                with cols[idx % num_cols]:
                    cached_data = st.session_state["cached_images"][file_key]
                    st.image(cached_data["image"], caption=cached_data["name"], use_container_width=True)
    else:
        # Xóa cache khi không có file
        if st.session_state["cached_images"]:
            st.session_state["cached_images"] = {}
        if st.session_state.get("file_contents"):
            st.session_state["file_contents"] = {}
    
    # Xác định ảnh đã chọn (đặt ở ngoài để dùng ở cả 2 cột)
    selected_image = None
    selected_image_name = None
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Chọn ảnh để phân tích
        if uploaded_files and st.session_state["cached_images"]:
            image_names = [data["name"] for data in st.session_state["cached_images"].values()]
            selected_image_name = st.selectbox(
                "Chọn ảnh để phân tích:",
                options=image_names,
                index=0
            )
            
            # Tìm ảnh đã chọn
            for key, data in st.session_state["cached_images"].items():
                if data["name"] == selected_image_name:
                    selected_image = data["image"]
                    break
            
            if selected_image:
                st.image(selected_image, caption=f"Ảnh đã chọn: {selected_image_name}", use_container_width=True)
        else:
            st.info("📤 Vui lòng tải ảnh lên trước")

    with col2:
        user_prompt = st.text_area(
            "Yêu cầu phân tích:", 
            value="Hãy mô tả chi tiết các bất thường trong ảnh và đưa ra chẩn đoán sơ bộ.",
            height=150
        )
        
        # Tùy chọn phân tích
        if uploaded_files and len(uploaded_files) > 1:
            analyze_mode = st.radio(
                "Chế độ phân tích:",
                ["Phân tích ảnh đã chọn", "Phân tích tất cả ảnh"],
                index=0
            )
        else:
            analyze_mode = "Phân tích ảnh đã chọn"
        
        if st.button("Phân tích / Tạo bệnh án", type="primary"):
            if not selected_image and analyze_mode == "Phân tích ảnh đã chọn":
                st.warning("⚠️ Vui lòng chọn ảnh để phân tích.")
            elif not st.session_state["cached_images"]:
                st.warning("⚠️ Vui lòng tải ảnh lên trước.")
            else:
                try:
                    images_to_analyze = []
                    
                    if analyze_mode == "Phân tích ảnh đã chọn":
                        images_to_analyze = [(selected_image_name, selected_image)]
                    else:
                        # Phân tích tất cả ảnh
                        images_to_analyze = [
                            (data["name"], data["image"]) 
                            for data in st.session_state["cached_images"].values()
                        ]
                    
                    # Phân tích từng ảnh
                    results = []
                    total = len(images_to_analyze)
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, (img_name, img) in enumerate(images_to_analyze):
                        status_text.text(f"🤖 Đang phân tích {img_name} ({idx+1}/{total})...")
                        
                        # Tìm file content tương ứng
                        file_content = None
                        for key, data in st.session_state["cached_images"].items():
                            if data["name"] == img_name:
                                file_key = key
                                file_content = st.session_state["file_contents"].get(file_key)
                                break
                        
                        # Kiểm tra cache trước
                        cached_result = None
                        if file_content:
                            cached_result = get_cached_result(file_content, user_prompt)
                        
                        if cached_result:
                            # Dùng kết quả từ cache (đã được clean khi lưu)
                            status_text.text(f"⚡ Đang load từ cache: {img_name} ({idx+1}/{total})...")
                            result = cached_result
                            # Clean lại để đảm bảo (nếu cache cũ chưa được clean)
                            result = clean_model_output(result, user_prompt)
                            results.append((img_name, result))
                        else:
                            # Generate mới
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
                            
                            # Xử lý ảnh và text
                            inputs = processor(text=prompt, images=img, return_tensors="pt")
                            
                            # Chuyển sang device phù hợp
                            device = next(model.parameters()).device
                            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                            # Generate
                            with torch.inference_mode():
                                out_ids = model.generate(
                                    **inputs, 
                                    max_new_tokens=128,
                                    do_sample=False,
                                    num_beams=1,
                                    pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
                                )
                            
                            # Decode chỉ phần response mới (không bao gồm input)
                            input_length = inputs["input_ids"].shape[1]
                            generated_ids = out_ids[0][input_length:]  # Chỉ lấy phần mới generate
                            full_result = processor.decode(generated_ids, skip_special_tokens=True)
                            
                            # Làm sạch kết quả: loại bỏ các phần không cần thiết
                            result = clean_model_output(full_result, user_prompt)
                            
                            # Lưu vào cache (lưu kết quả đã clean)
                            if file_content:
                                save_result(file_content, user_prompt, result, img_name)
                            
                            results.append((img_name, result))
                            
                            # Dọn dẹp RAM (chỉ khi generate, không phải load từ cache)
                            del inputs
                            del out_ids
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        # Cập nhật progress
                        progress_bar.progress((idx + 1) / total)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Hiển thị kết quả
                    st.success(f"✅ Đã phân tích {total} ảnh xong!")
                    st.markdown("### 📋 Kết quả phân tích:")
                    
                    for img_name, result in results:
                        with st.expander(f"📄 {img_name}", expanded=(len(results) == 1)):
                            st.write(result)
                            st.divider()
                    
                except Exception as e:
                    st.error(f"❌ Lỗi khi phân tích: {e}")
                    # Dọn dẹp khi lỗi
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()



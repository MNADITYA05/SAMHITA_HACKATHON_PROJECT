from __future__ import annotations
import os
import io
import time
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

from src.config import settings
from src.models import AnalysisResult
from src.analyzer import AgeGenderAnalyzer, calculate_statistics, format_processing_time
from src.image import validate_image, image_to_rgb_array, resize_for_display, draw_analysis_results


PAGE_CONFIG = {
    "page_title": "Age & Gender Detection System",
    "page_icon": "👥",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

if "analyzer" not in st.session_state:
    st.session_state.analyzer = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


@st.cache_resource
def load_analyzer() -> AgeGenderAnalyzer:
    with st.spinner("Loading AI models..."):
        try:
            return AgeGenderAnalyzer.from_defaults()
        except Exception as e:
            st.error(f"Failed to load AI models: {e}")
            return AgeGenderAnalyzer(face_app=None)


def main() -> None:
    st.set_page_config(**PAGE_CONFIG)
    st.title("👥 Age & Gender Detection System")
    st.markdown("### *Powered by InsightFace + DeepFace AI Models*")
    st.markdown("---")

    analyzer = load_analyzer()
    if not analyzer.model_loaded:
        st.error("⚠️ AI models failed to load. Please refresh the page or check your installation.")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        st.subheader("🎯 Detection")
        detection_confidence = st.slider("Detection Confidence", 0.1, 1.0, settings.detection_confidence, 0.05)
        min_face_size = st.slider("Minimum Face Size", 20, 100, settings.min_face_size, 5)

        st.subheader("🎨 Display Options")
        show_dual = st.checkbox("Show Both Model Results", value=True)
        show_emotion = st.checkbox("Show Emotion Analysis", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_processing_time = st.checkbox("Show Processing Time", value=settings.enable_performance_monitoring)

        with st.expander("🔧 Advanced Settings"):
            st.checkbox("Enable GPU", value=settings.enable_gpu, disabled=True)
            st.checkbox("Debug Mode", value=settings.debug_mode, disabled=True)

    tab1, tab2, tab3 = st.tabs(["📸 Analysis", "📊 Statistics", "ℹ️ Information"])

    with tab1:
        handle_analysis(analyzer, detection_confidence, min_face_size, show_dual, show_emotion, show_confidence, show_processing_time)

    with tab2:
        handle_statistics()

    with tab3:
        handle_information()


def handle_analysis(
    analyzer: AgeGenderAnalyzer,
    detection_confidence: float,
    min_face_size: int,
    show_dual: bool,
    show_emotion: bool,
    show_confidence: bool,
    show_processing_time: bool,
) -> None:
    st.header("📸 Face Analysis")

    input_method = st.radio("Choose Input Method:", ["📤 Upload Image", "📷 Camera Capture"], horizontal=True)

    uploaded_file = None
    camera_image = None
    if input_method == "📤 Upload Image":
        uploaded_file = st.file_uploader("Upload an image file", type=settings.supported_formats)
    else:
        camera_image = st.camera_input("Take a picture for analysis")

    image_source = uploaded_file or camera_image

    if image_source is None:
        st.info("👆 Please upload an image or take a photo to start analysis")
        if st.checkbox("Show Example Analysis"):
            show_example()
        return

    is_valid, msg = validate_image(Image.open(image_source))
    if not is_valid:
        st.error(f"❌ {msg}")
        return

    image = Image.open(image_source)
    image_array = image_to_rgb_array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(resize_for_display(image), caption="Input Image", use_container_width=True)
        st.info(f"📐 Dimensions: {image.size[0]} × {image.size[1]} pixels")

    with col2:
        st.subheader("🎯 Analysis Results")
        with st.spinner("🧠 Analyzing faces..."):
            t0 = time.time()
            results = analyzer.analyze_image(image_array, detection_confidence, min_face_size)
            elapsed = time.time() - t0

        if not results:
            st.info("ℹ️ No faces detected in the image")
            return

        filtered = [r for r in results if r.bbox.width >= min_face_size and r.bbox.height >= min_face_size]
        if not filtered:
            st.warning(f"⚠️ No faces detected above minimum size threshold ({min_face_size}px)")
            return

        annotated = draw_analysis_results(image_array.copy(), filtered, show_dual, show_emotion, show_confidence)
        annotated_pil = Image.fromarray(annotated.astype(np.uint8))
        st.image(resize_for_display(annotated_pil), caption="Analysis Results", use_container_width=True)

        if show_processing_time:
            st.success(f"⚡ Analysis completed in {format_processing_time(elapsed)}")

        st.session_state.analysis_history.append({
            "timestamp": datetime.now(),
            "results": filtered,
            "image_size": image.size,
            "processing_time": elapsed,
        })

        show_detailed_results(filtered, show_dual, show_emotion, show_confidence, show_processing_time)

        if st.button("💾 Download Annotated Image", key="download_btn"):
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                label="💾 Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"age_gender_analysis_{datetime.now():%Y%m%d_%H%M%S}.png",
                mime="image/png",
            )


def show_detailed_results(
    results: list[AnalysisResult],
    show_dual: bool,
    show_emotion: bool,
    show_confidence: bool,
    show_processing_time: bool,
) -> None:
    st.subheader("🔍 Detailed Analysis")
    stats = calculate_statistics(results)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("👥 Total Faces", stats.total_faces)
    with c2:
        st.metric("📊 Average Age", f"{stats.avg_age:.0f} years" if stats.avg_age > 0 else "N/A")
    with c3:
        st.metric("👨 Males", stats.male_count)
    with c4:
        st.metric("👩 Females", stats.female_count)

    for i, r in enumerate(results, 1):
        with st.expander(f"👤 Face {i} - {r.gender_final}, {r.age_final} years", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**🎂 Age Analysis**")
                if show_dual and r.age_deepface is not None:
                    st.write(f"• InsightFace: {r.age_insightface} years")
                    st.write(f"• DeepFace: {r.age_deepface} years")
                    st.write(f"• **Final: {r.age_final} years**")
                else:
                    st.write(f"• **Age: {r.age_final} years**")
                st.write(f"• **Group: {r.age_group}**")
            with c2:
                st.write("**👤 Gender Analysis**")
                if show_dual:
                    st.write(f"• InsightFace: {r.gender_insightface}")
                    st.write(f"• DeepFace: {r.gender_deepface}")
                    st.write(f"• **Final: {r.gender_final}**")
                else:
                    st.write(f"• **Gender: {r.gender_final}**")
                if show_confidence and r.gender_confidence > 0:
                    st.write(f"• Confidence: {r.gender_confidence:.1%}")
            with c3:
                st.write("**📈 Detection Info**")
                if show_confidence:
                    st.write(f"• Detection: {r.detection_score:.1%}")
                st.write(f"• Size: {r.bbox.width}×{r.bbox.height}px")
                if show_emotion and r.emotion.name != "UNKNOWN":
                    st.write(f"• **Emotion: {r.emotion}**")
                if show_processing_time:
                    st.write(f"• Processing: {format_processing_time(r.insightface_time + r.deepface_time)}")

            if show_emotion and r.emotion_scores:
                st.write("**😊 Emotion Breakdown**")
                df = pd.DataFrame([
                    {"Emotion": k.title(), "Score": v}
                    for k, v in r.emotion_scores.items()
                ]).sort_values("Score", ascending=False)
                if not df.empty:
                    fig = px.bar(df, x="Emotion", y="Score", title=f"Emotion Analysis - Face {i}", height=300)
                    st.plotly_chart(fig, use_container_width=True)


def handle_statistics() -> None:
    st.header("📊 Analysis Statistics")
    history = st.session_state.analysis_history
    if not history:
        st.info("📈 No analysis data yet. Analyze some images to see statistics!")
        return

    total_analyses = len(history)
    total_faces = sum(len(e["results"]) for e in history)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🖼️ Images Analyzed", total_analyses)
    with c2:
        st.metric("👥 Total Faces", total_faces)
    with c3:
        st.metric("📊 Avg Faces/Image", f"{total_faces / total_analyses:.1f}")
    with c4:
        st.metric("⏱️ Total Processing", format_processing_time(sum(e["processing_time"] for e in history)))

    all_results = [r for e in history for r in e["results"]]
    if not all_results:
        return

    st.subheader("👥 Demographics Distribution")
    c1, c2 = st.columns(2)

    ages = [r.age_final for r in all_results]
    with c1:
        fig = px.histogram(ages, title="Age Distribution", nbins=20, labels={"value": "Age", "count": "Count"}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        genders = pd.Series([str(r.gender_final) for r in all_results]).value_counts()
        fig = px.pie(values=genders.values, names=genders.index, title="Gender Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("👶 Age Groups Distribution")
    groups = pd.Series([str(r.age_group) for r in all_results]).value_counts()
    fig = px.bar(x=groups.index, y=groups.values, title="Distribution by Age Groups",
                 labels={"x": "Age Group", "y": "Count"}, height=400)
    st.plotly_chart(fig, use_container_width=True)

    emotions = [str(r.emotion) for r in all_results if r.emotion.name != "UNKNOWN"]
    if emotions:
        st.subheader("😊 Emotion Analysis")
        emo = pd.Series(emotions).value_counts()
        fig = px.bar(x=emo.index, y=emo.values, title="Detected Emotions Distribution",
                     labels={"x": "Emotion", "y": "Count"}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Analysis History")
    rows = []
    for i, entry in enumerate(history[-10:], max(1, len(history) - 9)):
        s = calculate_statistics(entry["results"])
        rows.append({
            "Analysis #": i,
            "Timestamp": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            "Faces Detected": s.total_faces,
            "Avg Age": f"{s.avg_age:.0f}" if s.avg_age > 0 else "N/A",
            "Males": s.male_count,
            "Females": s.female_count,
            "Processing Time": format_processing_time(entry["processing_time"]),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.button("🗑️ Clear Analysis History"):
        st.session_state.analysis_history = []
        st.success("✅ Analysis history cleared!")
        st.rerun()


def handle_information() -> None:
    st.header("ℹ️ System Information")

    st.subheader("🔬 How It Works")
    st.write("""
    This system uses two powerful AI models to analyze faces in images:

    **🎯 InsightFace:** Fast face detection and basic age/gender estimation, real-time processing, high accuracy.
    **🧠 DeepFace:** Advanced age estimation, precise gender classification, emotion recognition.
    **🔄 Combined Approach:** Both models for comparison and validation, confidence scores, dual-model results.
    """)

    st.subheader("✨ Key Features")
    c1, c2 = st.columns(2)
    with c1:
        st.write("""
        **🎯 Detection:**
        - Multi-face detection, adjustable confidence, min face size, camera support
        **📊 Analysis:**
        - Age estimation (0-100), gender classification, age groups, emotion recognition
        """)
    with c2:
        st.write("""
        **📈 Performance:**
        - Processing time monitoring, confidence scoring, statistical reporting
        **🎨 Display:**
        - Dual-model comparison, interactive visualizations, downloadable results, history
        """)

    st.subheader("📝 Usage Instructions")
    with st.expander("🚀 Getting Started"):
        st.write("1. **Choose Input Method**: Upload an image or use your camera\n"
                 "2. **Adjust Settings**: Use the sidebar to configure detection\n"
                 "3. **Analyze**: The system will automatically detect and analyze faces\n"
                 "4. **Review Results**: Check detailed analysis for each detected face\n"
                 "5. **View Statistics**: Monitor your analysis history and trends")

    with st.expander("⚙️ Settings Guide"):
        st.write("**Detection Confidence:** Higher values = more strict detection\n"
                 "**Minimum Face Size:** Filters out very small faces\n"
                 "**Display Options:** Dual Results, Emotion Analysis, Confidence Scores")

    st.subheader("🔧 Technical Specifications")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Supported:** JPEG, PNG, BMP, TIFF (50×50 to 5000×5000 px)")
    with c2:
        st.write("**Detection:** Multiple faces, 0-100 years, Male/Female, 7 emotions")

    st.subheader("🤖 AI Models Information")
    with st.expander("📚 Model Details"):
        st.write("**InsightFace:** ResNet-based CNN, 5M+ faces, ~95% age / ~98% gender, ~100ms\n"
                 "**DeepFace:** VGG-Face, diverse demographics, ~97% age / ~99% gender, age/gender/emotion")

    st.subheader("❓ FAQ")
    with st.expander("Why do I get different results from both models?"):
        st.write("Different training data and algorithms. Showing both lets you compare.")
    with st.expander("How accurate are the age predictions?"):
        st.write("±3-5 years for adults, ±2-3 years for children. Depends on image quality.")
    with st.expander("What if no faces are detected?"):
        st.write("Lower the confidence threshold, ensure good lighting and front-facing faces.")


def show_example() -> None:
    st.subheader("📖 Example Analysis")
    data = {
        "Face 1": {"Age": 28, "Gender": "Female", "Emotion": "Happy", "Confidence": "94%"},
        "Face 2": {"Age": 35, "Gender": "Male", "Emotion": "Neutral", "Confidence": "91%"},
        "Face 3": {"Age": 12, "Gender": "Female", "Emotion": "Surprise", "Confidence": "89%"},
    }
    for face, d in data.items():
        with st.expander(f"👤 {face} - {d['Gender']}, {d['Age']} years"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**Age:** {d['Age']} years")
            with c2:
                st.write(f"**Gender:** {d['Gender']}")
            with c3:
                st.write(f"**Emotion:** {d['Emotion']}")
            st.write(f"**Detection Confidence:** {d['Confidence']}")


if __name__ == "__main__":
    os.makedirs(settings.temp_dir, exist_ok=True)
    os.makedirs(settings.models_dir, exist_ok=True)
    main()

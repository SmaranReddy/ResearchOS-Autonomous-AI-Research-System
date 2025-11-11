import streamlit as st
import os
import sys
import io
import json
from contextlib import redirect_stdout

# ---------------------------------------------------------------------
# 🧩 Fix Import Path (important)
# ---------------------------------------------------------------------
# Add the project root (parent of frontend/) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from backend/
from backend.pipeline import main as run_pipeline


# ---------------------------------------------------------------------
# 🎨 Streamlit Page Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Tavily Research Assistant",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------
# 🧠 Header Section
# ---------------------------------------------------------------------
st.title("🧠 Tavily Research Assistant")
st.markdown("""
Welcome to your **AI-Powered Research Pipeline**!  
This app automates your research process by:
1. Searching for papers via Tavily  
2. Downloading and extracting text  
3. Cleaning, tokenizing, and chunking content  
4. Summarizing key sections  
5. Embedding and indexing results for retrieval

---
""")

# ---------------------------------------------------------------------
# 🔍 User Input
# ---------------------------------------------------------------------
query = st.text_input(
    "Enter your research query:",
    placeholder="e.g. Graph neural networks in molecular biology"
)

# ---------------------------------------------------------------------
# 🚀 Run Button
# ---------------------------------------------------------------------
if st.button("🚀 Run Research Pipeline", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a query first.")
        st.stop()

    # Display spinner and capture output
    with st.spinner("Running Tavily Research Pipeline... Please wait ⏳"):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            try:
                run_pipeline(query)
            except Exception as e:
                st.error(f"Pipeline execution failed: {e}")

    # -----------------------------------------------------------------
    # 🧾 Show Logs
    # -----------------------------------------------------------------
    st.subheader("🧾 Pipeline Execution Logs")
    st.text_area("Execution Log", buffer.getvalue(), height=350)

    # -----------------------------------------------------------------
    # 📂 Load Output JSON (if exists)
    # -----------------------------------------------------------------
    safe_name = query.replace(" ", "_").replace("/", "_")
    json_path = os.path.join("downloads", f"{safe_name}.json")

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            paper_data = json.load(f)

        st.success("✅ Research pipeline completed successfully!")

        # -------------------------------------------------------------
        # 📘 Paper Details
        # -------------------------------------------------------------
        st.subheader("📘 Paper Details")
        st.write(f"**Title:** {paper_data.get('title', 'N/A')}")
        st.write(f"**Link:** [{paper_data.get('link', 'N/A')}]({paper_data.get('link', '#')})")

        # -------------------------------------------------------------
        # 📝 Summary
        # -------------------------------------------------------------
        if "summary_structured" in paper_data:
            st.subheader("📝 Summary")
            st.write(paper_data["summary_structured"])

        # -------------------------------------------------------------
        # 📄 Extracted Full Text
        # -------------------------------------------------------------
        with st.expander("📄 View Extracted Full Text"):
            st.text_area(
                "Full Text",
                paper_data.get("full_text", "No extracted text available."),
                height=300
            )

        # -------------------------------------------------------------
        # 💾 Download Button
        # -------------------------------------------------------------
        st.download_button(
            label="💾 Download Paper JSON",
            data=json.dumps(paper_data, ensure_ascii=False, indent=2),
            file_name=f"{safe_name}.json",
            mime="application/json",
        )
    else:
        st.warning("⚠️ No paper JSON found. The pipeline may have stopped early.")

# ---------------------------------------------------------------------
# 📚 Sidebar
# ---------------------------------------------------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
**Tavily AI Research Assistant**  
Built with:
- 🧠 Python + Streamlit  
- 🔍 Tavily API  
- 🧩 Pinecone + LangChain-style multi-agent pipeline  

Use this app to automate literature search, summarization, and indexing.  
Ideal for students, researchers, and AI developers.
""")

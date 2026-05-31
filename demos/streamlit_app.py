import json
import tempfile

import streamlit as st

from legal_pdf_extractor.api import extract

st.set_page_config(page_title="Legal PDF Extractor", layout="wide")
st.title("Legal PDF Extractor")

uploaded_pdf = st.file_uploader("PDF", type=["pdf"])
query = st.text_input("Query", placeholder="Who is the tenant?")
output_type = st.selectbox(
    "Output type",
    ["string", "date", "number", "array[string]", "array[date]", "array[number]"],
)
examples_json = st.text_area(
    "Few-shot examples(Optional)",
    placeholder=(
        '[{"input": "Tenant: Acme LLC.", '
        '"output": {"value": "Acme LLC", "found": true, "sources": []}}]'
    ),
    height=120,
)

if uploaded_pdf and query and st.button("Extract"):
    examples = None
    if examples_json.strip():
        try:
            examples = json.loads(examples_json)
            if not isinstance(examples, list):
                st.error("Examples must be a JSON array.")
                st.stop()
        except json.JSONDecodeError as exc:
            st.error(f"Invalid examples JSON: {exc.msg}")
            st.stop()

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.getvalue())
        tmp.flush()
        result = extract(tmp.name, query=query, output_type=output_type, examples=examples)
    st.json(result)

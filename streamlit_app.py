from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from smolagents import OpenAIModel, ToolCallingAgent

import agent
import pipeline
import tools.dataset as dataset


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
INSTRUCTIONS_PATH = PROJECT_ROOT / "prompts" / "agent_instructions.txt"


# ============================================================
# Streamlit page setup
# ============================================================

st.set_page_config(
    page_title="ChemASSistant",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 ChemASSistant")
st.caption(
    "Demo version — do not enter confidential or sensitive data."
)


# ============================================================
# Agent setup
# ============================================================

@st.cache_resource
def create_agent() -> ToolCallingAgent:
    """Create and cache the ChemASSistant agent."""

    instructions = INSTRUCTIONS_PATH.read_text(encoding="utf-8")

    model = OpenAIModel(
        model_id="gemini-3.5-flash-lite",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=st.secrets["GEMINI_API_KEY"],
    )

    return ToolCallingAgent(
        tools=[
            agent.validate_molecules_tool,
            agent.calculate_descriptors_tool,
            agent.check_property_rules_tool,
            agent.find_similar_molecules_tool,
            agent.search_reference_database_tool,
            agent.get_reference_molecule_tool,
            agent.find_similar_in_reference_database_tool,
        ],
        model=model,
        instructions=instructions,
    )


# ============================================================
# Main UI
# ============================================================

chat_tab, dataset_tab = st.tabs(
    ["💬 ChemASSistant", "📄 Dataset analysis"]
)


# ============================================================
# Chat tab
# ============================================================

with chat_tab:

    st.subheader("Molecular analysis assistant")

    st.markdown(
        """
        Example questions:

        - `Validate these SMILES: CCO, CCN, C1CC`
        - `Calculate descriptors for aspirin: CC(=O)Oc1ccccc1C(=O)O`
        - `Find aspirin in the local reference database`
        - `Find the 3 reference molecules most similar to O=C=O`
        """
    )

    prompt = st.text_area(
        "Ask ChemASSistant",
        height=180,
        placeholder="Enter a cheminformatics question...",
    )

    if st.button(
        "Analyze",
        type="primary",
        key="chat_analyze",
    ):
        if not prompt.strip():
            st.warning("Enter a question first.")
        else:
            try:
                chem_agent = create_agent()

                with st.spinner("Analyzing..."):
                    response = chem_agent.run(prompt)

                st.markdown("### Response")
                st.write(response)

            except Exception as exc:
                st.error(f"ChemASSistant error: {exc}")


# ============================================================
# Dataset analysis tab
# ============================================================

with dataset_tab:

    st.subheader("CSV molecular dataset")

    st.write(
        "Upload a CSV file containing a column with molecular SMILES. "
        "The analysis below is deterministic and runs without the LLM."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="dataset_upload",
    )

    if uploaded_file is not None:

        try:
            raw_df = pd.read_csv(uploaded_file)

        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            raw_df = None

        if raw_df is not None:

            if raw_df.empty:
                st.warning("The uploaded CSV contains no data rows.")

            elif len(raw_df.columns) == 0:
                st.warning("The uploaded CSV contains no columns.")

            else:
                st.markdown("### Dataset preview")
                st.dataframe(
                    raw_df.head(20),
                    use_container_width=True,
                )

                detected_column = dataset.detect_smiles_column(raw_df)

                column_names = list(raw_df.columns)

                if detected_column in column_names:
                    default_index = column_names.index(detected_column)
                else:
                    default_index = 0

                smiles_column = st.selectbox(
                    "SMILES column",
                    options=column_names,
                    index=default_index,
                    help=(
                        "ChemASSistant tries to detect a SMILES column "
                        "automatically, but you can override it here."
                    ),
                )

                if detected_column is not None:
                    st.caption(
                        f"Automatically detected SMILES column: "
                        f"`{detected_column}`"
                    )
                else:
                    st.caption(
                        "No SMILES column was detected automatically. "
                        "Select the correct column manually."
                    )

                if st.button(
                    "Analyze dataset",
                    type="primary",
                    key="dataset_analyze",
                ):

                    try:
                        prepared_df = dataset.prepare_dataset(
                            raw_df,
                            smiles_column=smiles_column,
                        )

                        result = pipeline.analyze_dataframe(
                            prepared_df,
                            smiles_column="canonical_smiles",
                        )

                        analyzed_df = result["dataset"]
                        validation_summary = result["validation_summary"]

                        st.markdown("### Summary")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Total molecules",
                            validation_summary["total_molecules"],
                        )

                        col2.metric(
                            "Valid molecules",
                            validation_summary["valid_molecules"],
                        )

                        col3.metric(
                            "Invalid molecules",
                            validation_summary["invalid_molecules"],
                        )

                        st.markdown("### Analyzed dataset")

                        st.dataframe(
                            analyzed_df,
                            use_container_width=True,
                        )

                        st.markdown("### Top similar pairs")

                        top_pairs = result["top_similar_pairs"]

                        if top_pairs:
                            similarity_df = pd.DataFrame(top_pairs)

                            st.dataframe(
                                similarity_df,
                                use_container_width=True,
                            )
                        else:
                            st.info(
                                "Not enough valid molecules to calculate "
                                "pairwise similarities."
                            )

                        csv_output = analyzed_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "Download analyzed CSV",
                            data=csv_output,
                            file_name="chemassistant_analysis.csv",
                            mime="text/csv",
                        )

                    except Exception as exc:
                        st.error(f"Dataset analysis failed: {exc}")

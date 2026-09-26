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
# Language
# ============================================================

if "ui_language" not in st.session_state:
    st.session_state.ui_language = "hr"

TEXT = {
    "hr": {
        "page_title": "ChemASSistant",
        "caption": "Demo verzija — nemojte unositi povjerljive ili osjetljive podatke.",
        "switch_button": "EN",
        "switch_help": "Prebaci sučelje na engleski",
        "chat_tab": "💬 ChemASSistant",
        "dataset_tab": "📄 Analiza skupa podataka",
        "chat_header": "Asistent za molekularnu analizu",
        "examples": """
Primjeri pitanja:

- `Validiraj ove SMILES zapise: CCO, CCN, C1CC`
- `Izračunaj deskriptore za aspirin: CC(=O)Oc1ccccc1C(=O)O`
- `Pronađi aspirin u lokalnoj referentnoj bazi`
- `Pronađi 3 referentne molekule najsličnije O=C=O`
""",
        "ask_label": "Pitaj ChemASSistant",
        "ask_placeholder": "Unesite pitanje iz područja kemoinformatike...",
        "analyze_button": "Analiziraj",
        "enter_question": "Prvo unesite pitanje.",
        "analyzing": "Analiziram...",
        "response": "Odgovor",
        "assistant_error": "Greška ChemASSistant-a",
        "dataset_header": "CSV skup molekularnih podataka",
        "dataset_intro": (
            "Prenesite CSV datoteku koja sadrži stupac s molekularnim SMILES zapisima. "
            "Analiza u ovom odjeljku je deterministička i ne koristi LLM."
        ),
        "upload_csv": "Prenesi CSV",
        "csv_read_error": "Nije moguće pročitati CSV",
        "empty_rows": "Preneseni CSV ne sadrži retke podataka.",
        "empty_columns": "Preneseni CSV ne sadrži stupce.",
        "dataset_preview": "Pregled skupa podataka",
        "smiles_column": "SMILES stupac",
        "smiles_help": (
            "ChemASSistant pokušava automatski prepoznati SMILES stupac, "
            "ali ga ovdje možete ručno promijeniti."
        ),
        "detected_smiles": "Automatski prepoznat SMILES stupac",
        "not_detected": (
            "SMILES stupac nije automatski prepoznat. "
            "Ručno odaberite odgovarajući stupac."
        ),
        "analyze_dataset": "Analiziraj skup podataka",
        "summary": "Sažetak",
        "total_molecules": "Ukupno molekula",
        "valid_molecules": "Valjane molekule",
        "invalid_molecules": "Nevaljane molekule",
        "analyzed_dataset": "Analizirani skup podataka",
        "top_pairs": "Najsličniji parovi",
        "not_enough": "Nema dovoljno valjanih molekula za izračun parnih sličnosti.",
        "download_csv": "Preuzmi analizirani CSV",
        "dataset_error": "Analiza skupa podataka nije uspjela",
    },
    "en": {
        "page_title": "ChemASSistant",
        "caption": "Demo version — do not enter confidential or sensitive data.",
        "switch_button": "HR",
        "switch_help": "Switch interface to Croatian",
        "chat_tab": "💬 ChemASSistant",
        "dataset_tab": "📄 Dataset analysis",
        "chat_header": "Molecular analysis assistant",
        "examples": """
Example questions:

- `Validate these SMILES: CCO, CCN, C1CC`
- `Calculate descriptors for aspirin: CC(=O)Oc1ccccc1C(=O)O`
- `Find aspirin in the local reference database`
- `Find the 3 reference molecules most similar to O=C=O`
""",
        "ask_label": "Ask ChemASSistant",
        "ask_placeholder": "Enter a cheminformatics question...",
        "analyze_button": "Analyze",
        "enter_question": "Enter a question first.",
        "analyzing": "Analyzing...",
        "response": "Response",
        "assistant_error": "ChemASSistant error",
        "dataset_header": "CSV molecular dataset",
        "dataset_intro": (
            "Upload a CSV file containing a column with molecular SMILES. "
            "The analysis in this section is deterministic and does not use the LLM."
        ),
        "upload_csv": "Upload CSV",
        "csv_read_error": "Could not read CSV",
        "empty_rows": "The uploaded CSV contains no data rows.",
        "empty_columns": "The uploaded CSV contains no columns.",
        "dataset_preview": "Dataset preview",
        "smiles_column": "SMILES column",
        "smiles_help": (
            "ChemASSistant tries to detect a SMILES column automatically, "
            "but you can override it here."
        ),
        "detected_smiles": "Automatically detected SMILES column",
        "not_detected": (
            "No SMILES column was detected automatically. "
            "Select the correct column manually."
        ),
        "analyze_dataset": "Analyze dataset",
        "summary": "Summary",
        "total_molecules": "Total molecules",
        "valid_molecules": "Valid molecules",
        "invalid_molecules": "Invalid molecules",
        "analyzed_dataset": "Analyzed dataset",
        "top_pairs": "Top similar pairs",
        "not_enough": "Not enough valid molecules to calculate pairwise similarities.",
        "download_csv": "Download analyzed CSV",
        "dataset_error": "Dataset analysis failed",
    },
}

lang = st.session_state.ui_language
t = TEXT[lang]


# ============================================================
# Streamlit page setup
# ============================================================

st.set_page_config(
    page_title=t["page_title"],
    page_icon="🧪",
    layout="wide",
)

title_col, lang_col = st.columns([10, 1])

with title_col:
    st.title("🧪 ChemASSistant")
    st.caption(t["caption"])

with lang_col:
    if st.button(
        t["switch_button"],
        help=t["switch_help"],
        key="language_toggle",
        use_container_width=True,
    ):
        st.session_state.ui_language = "en" if lang == "hr" else "hr"
        st.rerun()


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
    [t["chat_tab"], t["dataset_tab"]]
)


# ============================================================
# Chat tab
# ============================================================

with chat_tab:

    st.subheader(t["chat_header"])
    st.markdown(t["examples"])

    prompt = st.text_area(
        t["ask_label"],
        height=180,
        placeholder=t["ask_placeholder"],
    )

    if st.button(
        t["analyze_button"],
        type="primary",
        key="chat_analyze",
    ):
        if not prompt.strip():
            st.warning(t["enter_question"])
        else:
            try:
                chem_agent = create_agent()

                with st.spinner(t["analyzing"]):
                    response = chem_agent.run(prompt)

                st.markdown(f"### {t['response']}")
                st.write(response)

            except Exception as exc:
                st.error(f"{t['assistant_error']}: {exc}")


# ============================================================
# Dataset analysis tab
# ============================================================

with dataset_tab:

    st.subheader(t["dataset_header"])
    st.write(t["dataset_intro"])

    uploaded_file = st.file_uploader(
        t["upload_csv"],
        type=["csv"],
        key="dataset_upload",
    )

    if uploaded_file is not None:

        try:
            raw_df = pd.read_csv(uploaded_file)

        except Exception as exc:
            st.error(f"{t['csv_read_error']}: {exc}")
            raw_df = None

        if raw_df is not None:

            if raw_df.empty:
                st.warning(t["empty_rows"])

            elif len(raw_df.columns) == 0:
                st.warning(t["empty_columns"])

            else:
                st.markdown(f"### {t['dataset_preview']}")
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
                    t["smiles_column"],
                    options=column_names,
                    index=default_index,
                    help=t["smiles_help"],
                )

                if detected_column is not None:
                    st.caption(
                        f"{t['detected_smiles']}: `{detected_column}`"
                    )
                else:
                    st.caption(t["not_detected"])

                if st.button(
                    t["analyze_dataset"],
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

                        st.markdown(f"### {t['summary']}")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            t["total_molecules"],
                            validation_summary["total_molecules"],
                        )

                        col2.metric(
                            t["valid_molecules"],
                            validation_summary["valid_molecules"],
                        )

                        col3.metric(
                            t["invalid_molecules"],
                            validation_summary["invalid_molecules"],
                        )

                        st.markdown(f"### {t['analyzed_dataset']}")

                        st.dataframe(
                            analyzed_df,
                            use_container_width=True,
                        )

                        st.markdown(f"### {t['top_pairs']}")

                        top_pairs = result["top_similar_pairs"]

                        if top_pairs:
                            similarity_df = pd.DataFrame(top_pairs)

                            st.dataframe(
                                similarity_df,
                                use_container_width=True,
                            )
                        else:
                            st.info(t["not_enough"])

                        csv_output = analyzed_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            t["download_csv"],
                            data=csv_output,
                            file_name="chemassistant_analysis.csv",
                            mime="text/csv",
                        )

                    except Exception as exc:
                        st.error(f"{t['dataset_error']}: {exc}")

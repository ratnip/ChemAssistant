from __future__ import annotations

import hashlib
import io
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
# Session state
# ============================================================

DEFAULT_STATE = {
    "ui_language": "hr",
    "dataset_context": None,
    "dataset_filename": None,
    "dataset_file_hash": None,
    "analyzed_df": None,
    "analysis_result": None,
    "reference_compare_enabled": False,
    "messages": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Language strings
# ============================================================

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
- `Pronađi aspirin u lokalnoj referentnoj bazi`
- `Koja molekula u učitanom CSV-u ima najveći MW?`
- `Koji je najsličniji par molekula iz učitanog CSV-a?`
- `Koje su tri referentne molekule najsličnije CMP001?`
""",
        "ask_label": "Pitaj ChemASSistant",
        "ask_placeholder": "Unesite pitanje iz područja kemoinformatike...",
        "analyze_button": "Analiziraj",
        "enter_question": "Prvo unesite pitanje.",
        "analyzing": "Analiziram...",
        "response": "Odgovor",
        "assistant_error": "Greška ChemASSistant-a",
        "dataset_active": "Aktivan CSV kontekst",
        "dataset_none": "CSV skup podataka trenutno nije povezan s chatom.",
        "clear_dataset": "Odspoji CSV od chata",
        "clear_chat": "Očisti razgovor",
        "dataset_header": "CSV skup molekularnih podataka",
        "dataset_intro": (
            "Prenesite CSV datoteku sa SMILES stupcem. Analiza je deterministička. "
            "Nakon analize, sažeti rezultati postaju dostupni chatu."
        ),
        "upload_csv": "Prenesi CSV",
        "csv_read_error": "Nije moguće pročitati CSV",
        "empty_rows": "Preneseni CSV ne sadrži retke podataka.",
        "empty_columns": "Preneseni CSV ne sadrži stupce.",
        "dataset_preview": "Pregled skupa podataka",
        "smiles_column": "SMILES stupac",
        "smiles_help": (
            "ChemASSistant pokušava automatski prepoznati SMILES stupac, "
            "ali ga možete ručno promijeniti."
        ),
        "detected_smiles": "Automatski prepoznat SMILES stupac",
        "not_detected": (
            "SMILES stupac nije automatski prepoznat. "
            "Ručno odaberite odgovarajući stupac."
        ),
        "compare_reference": "Usporedi s lokalnom ChEMBL referentnom bazom",
        "compare_reference_help": (
            "Za svaku valjanu molekulu Python/RDKit lokalno pronalazi "
            "3 najsličnije referentne molekule."
        ),
        "analyze_dataset": "Analiziraj skup podataka",
        "summary": "Sažetak",
        "total_molecules": "Ukupno molekula",
        "valid_molecules": "Valjane molekule",
        "invalid_molecules": "Nevaljane molekule",
        "analyzed_dataset": "Analizirani skup podataka",
        "top_pairs": "Najsličniji parovi unutar CSV-a",
        "reference_matches": "Sličnost prema referentnoj bazi",
        "not_enough": "Nema dovoljno valjanih molekula za izračun parnih sličnosti.",
        "download_csv": "Preuzmi analizirani CSV",
        "dataset_error": "Analiza skupa podataka nije uspjela",
        "chat_connected": "CSV analiza je sada dostupna chatu.",
        "context_note": (
            "Chat dobiva deterministički sažetak i najviše 50 analiziranih redaka "
            "kako ne bismo nepotrebno trošili tokene."
        ),
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
- `Find aspirin in the local reference database`
- `Which molecule in the uploaded CSV has the highest MW?`
- `Which pair in the uploaded CSV is most similar?`
- `Which three reference molecules are most similar to CMP001?`
""",
        "ask_label": "Ask ChemASSistant",
        "ask_placeholder": "Enter a cheminformatics question...",
        "analyze_button": "Analyze",
        "enter_question": "Enter a question first.",
        "analyzing": "Analyzing...",
        "response": "Response",
        "assistant_error": "ChemASSistant error",
        "dataset_active": "Active CSV context",
        "dataset_none": "No CSV dataset is currently connected to chat.",
        "clear_dataset": "Disconnect CSV from chat",
        "clear_chat": "Clear conversation",
        "dataset_header": "CSV molecular dataset",
        "dataset_intro": (
            "Upload a CSV file containing a SMILES column. Analysis is deterministic. "
            "After analysis, compact results become available to chat."
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
        "compare_reference": "Compare with local ChEMBL reference database",
        "compare_reference_help": (
            "For each valid molecule, Python/RDKit locally finds "
            "the 3 most similar reference molecules."
        ),
        "analyze_dataset": "Analyze dataset",
        "summary": "Summary",
        "total_molecules": "Total molecules",
        "valid_molecules": "Valid molecules",
        "invalid_molecules": "Invalid molecules",
        "analyzed_dataset": "Analyzed dataset",
        "top_pairs": "Top similar pairs within CSV",
        "reference_matches": "Reference-database similarity",
        "not_enough": "Not enough valid molecules to calculate pairwise similarities.",
        "download_csv": "Download analyzed CSV",
        "dataset_error": "Dataset analysis failed",
        "chat_connected": "CSV analysis is now available to chat.",
        "context_note": (
            "Chat receives deterministic summary data plus at most 50 analyzed rows "
            "to avoid unnecessary token usage."
        ),
    },
}

lang = st.session_state.ui_language
t = TEXT[lang]


# ============================================================
# Helpers
# ============================================================

def build_dataset_context(
    analyzed_df: pd.DataFrame,
    result: dict,
    filename: str | None,
    max_rows: int = 50,
) -> str:
    """Create a compact deterministic dataset context for the LLM."""

    preferred_columns = [
        "compound_id",
        "id",
        "name",
        "original_smiles",
        "canonical_smiles",
        "valid",
        "validation_error",
        "mw",
        "logp",
        "tpsa",
        "hbd",
        "hba",
        "rotatable_bonds",
        "heavy_atoms",
        "formal_charge",
        "lipinski_violations",
        "violations",
        "ref_1_chembl_id",
        "ref_1_name",
        "ref_1_similarity",
        "ref_2_chembl_id",
        "ref_2_name",
        "ref_2_similarity",
        "ref_3_chembl_id",
        "ref_3_name",
        "ref_3_similarity",
    ]

    selected_columns = [
        column for column in preferred_columns
        if column in analyzed_df.columns
    ]

    # Preserve a few user-provided columns too, while keeping context bounded.
    for column in analyzed_df.columns:
        if column not in selected_columns and len(selected_columns) < 24:
            selected_columns.append(column)

    context_df = analyzed_df[selected_columns].head(max_rows).copy()
    context_df = context_df.where(pd.notna(context_df), None)

    truncated = len(analyzed_df) > max_rows

    return f"""
CURRENT ANALYZED DATASET CONTEXT

This context was produced by ChemASSistant's deterministic Python/RDKit
pipeline. Treat values in this context as authoritative dataset results.
Do not recalculate them from model knowledge.

Dataset file: {filename or "unknown"}
Total rows: {len(analyzed_df)}
Rows included below: {len(context_df)}
Context truncated: {truncated}

Validation summary:
{result["validation_summary"]}

Descriptor summary:
{result["descriptor_summary"]}

Top pairwise similarities within the uploaded dataset:
{result["top_similar_pairs"]}

Analyzed rows:
{context_df.to_dict(orient="records")}

IMPORTANT:
- Answer in the same language as the user's question.
- Use only the dataset information shown above for claims about the uploaded CSV.
- Reference-match columns were calculated locally with Morgan fingerprints
  (radius=2) and Tanimoto similarity.
- Do not describe similarity scores as percentages of chemical similarity.
- If the requested answer depends on rows not included because the context was
  truncated, say that the current chat context does not contain enough rows
  to answer reliably.
- Do not invent missing dataset values.
""".strip()


def clear_analysis_state() -> None:
    """Clear state associated with a previously uploaded/analyzed dataset."""
    st.session_state.dataset_context = None
    st.session_state.dataset_filename = None
    st.session_state.analyzed_df = None
    st.session_state.analysis_result = None
    st.session_state.reference_compare_enabled = False


def render_analysis_results() -> None:
    """Render persisted analysis results after reruns/language changes."""
    analyzed_df = st.session_state.analyzed_df
    result = st.session_state.analysis_result

    if analyzed_df is None or result is None:
        return

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
    st.dataframe(analyzed_df, use_container_width=True)

    st.markdown(f"### {t['top_pairs']}")
    top_pairs = result["top_similar_pairs"]

    if top_pairs:
        st.dataframe(
            pd.DataFrame(top_pairs),
            use_container_width=True,
        )
    else:
        st.info(t["not_enough"])

    reference_matches = result.get("reference_matches", [])

    if reference_matches:
        flattened = []

        for item in reference_matches:
            query_smiles = item["query_smiles"]

            for rank, match in enumerate(item["matches"], start=1):
                flattened.append(
                    {
                        "query_smiles": query_smiles,
                        "rank": rank,
                        "chembl_id": match.get("chembl_id"),
                        "name": match.get("name"),
                        "reference_smiles": match.get("smiles"),
                        "similarity": match.get("similarity"),
                    }
                )

        if flattened:
            st.markdown(f"### {t['reference_matches']}")
            st.dataframe(
                pd.DataFrame(flattened),
                use_container_width=True,
            )

    csv_output = analyzed_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        t["download_csv"],
        data=csv_output,
        file_name="chemassistant_analysis.csv",
        mime="text/csv",
    )


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
# Tabs
# ============================================================

chat_tab, dataset_tab = st.tabs(
    [t["chat_tab"], t["dataset_tab"]]
)


# ============================================================
# Dataset analysis tab
# Executed before chat so newly-created context is visible immediately.
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
        uploaded_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(uploaded_bytes).hexdigest()

        # If the actual file changes, discard stale analysis/context.
        if (
            st.session_state.dataset_file_hash is not None
            and st.session_state.dataset_file_hash != current_hash
        ):
            clear_analysis_state()

        st.session_state.dataset_file_hash = current_hash

        try:
            raw_df = pd.read_csv(io.BytesIO(uploaded_bytes))
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

                default_index = (
                    column_names.index(detected_column)
                    if detected_column in column_names
                    else 0
                )

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

                compare_reference = st.checkbox(
                    t["compare_reference"],
                    value=st.session_state.reference_compare_enabled,
                    help=t["compare_reference_help"],
                )

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
                            compare_reference=compare_reference,
                            reference_top_n=3,
                        )

                        analyzed_df = result["dataset"]

                        # Persist full deterministic analysis across Streamlit reruns.
                        st.session_state.analyzed_df = analyzed_df
                        st.session_state.analysis_result = result
                        st.session_state.dataset_filename = uploaded_file.name
                        st.session_state.reference_compare_enabled = compare_reference

                        # Persist compact context for the LLM.
                        st.session_state.dataset_context = build_dataset_context(
                            analyzed_df=analyzed_df,
                            result=result,
                            filename=uploaded_file.name,
                            max_rows=50,
                        )

                        st.success(t["chat_connected"])

                    except Exception as exc:
                        st.error(f"{t['dataset_error']}: {exc}")

                # Render results even after language changes or other reruns.
                render_analysis_results()


# ============================================================
# Chat tab
# ============================================================

with chat_tab:

    st.subheader(t["chat_header"])

    status_col, clear_col = st.columns([5, 1])

    with status_col:
        if st.session_state.dataset_context is not None:
            st.success(
                f"{t['dataset_active']}: "
                f"{st.session_state.dataset_filename or 'CSV'}"
            )
            st.caption(t["context_note"])
        else:
            st.info(t["dataset_none"])

    with clear_col:
        if st.button(
            t["clear_chat"],
            key="clear_chat_history",
            use_container_width=True,
        ):
            st.session_state.messages = []
            st.rerun()

    if st.session_state.dataset_context is not None:
        if st.button(
            t["clear_dataset"],
            key="clear_dataset_context",
        ):
            st.session_state.dataset_context = None
            st.session_state.dataset_filename = None
            st.rerun()

    st.markdown(t["examples"])

    # Show the complete visible conversation for this browser session.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.text_area(
        t["ask_label"],
        height=140,
        placeholder=t["ask_placeholder"],
        key="chat_prompt",
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

                # Build a bounded conversational context.
                # Keep only the previous 8 messages for LLM input,
                # while the UI can display the full in-session conversation.
                recent_messages = st.session_state.messages[-8:]

                history_lines = []
                for message in recent_messages:
                    role = "USER" if message["role"] == "user" else "ASSISTANT"
                    history_lines.append(
                        f"{role}: {message['content']}"
                    )

                agent_parts = []

                if st.session_state.dataset_context is not None:
                    agent_parts.append(st.session_state.dataset_context)

                if history_lines:
                    agent_parts.append(
                        "RECENT CONVERSATION:\n"
                        + "\n".join(history_lines)
                    )

                agent_parts.append(
                    "USER QUESTION:\n" + prompt.strip()
                )

                agent_prompt = "\n\n".join(agent_parts)

                # Add user message once, before the model call.
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                )

                with st.spinner(t["analyzing"]):
                    response = chem_agent.run(agent_prompt)

                response_text = str(response)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                    }
                )

                # Rerun so the new user + assistant messages are rendered
                # through the same chat history code above.
                st.rerun()

            except Exception as exc:
                st.error(f"{t['assistant_error']}: {exc}")

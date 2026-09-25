import streamlit as st
from smolagents import OpenAIModel, ToolCallingAgent
import agent

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

INSTRUCTIONS_PATH = PROJECT_ROOT / "prompts" / "agent_instructions.txt"

instructions = INSTRUCTIONS_PATH.read_text(
    encoding="utf-8"
)

st.set_page_config(
    page_title="ChemASSistant",
    page_icon="🧪",
    layout="wide",
)


model = OpenAIModel(
    model_id="gemini-3.5-flash-lite",
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=st.secrets["GEMINI_API_KEY"],
)





chem_agent = ToolCallingAgent(
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


st.title("🧪 ChemASSistant")
st.write("Agentni asistent za kemoinformatiku")

st.caption(
    "Postavi pitanje o molekulama, njihovim svojstvima, "
    "valjanosti struktura ili međusobnoj sličnosti."
)

prompt = st.text_area(
    "Upit",
    placeholder=(
        "Primjer:\n\n"
        "Pronađi aspirin u lokalnoj referentnoj bazi i reci mi njegov ChEMBL ID, molekulsku formulu i SMILES.\n\n"
        "ili: \n\n"
        "Koje su tri molekule u referentnoj bazi najsličnije CO2?\n\n"
        "ili: \n\n"
        "Imam sljedeće spojeve:\n"
        "CCO\n"
        "CCN\n"
        "CCCC\n"
        "CC(=O)OC1=CC=CC=C1C(=O)O\n\n"
        "Koja su dva para međusobno najsličnija i postoje limeđu ovim spojevima kršenja pravila?\n\n"
        
        
    ),
    height=400,
)

if st.button("Analiziraj", type="primary"):
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analiziram..."):
                response = chem_agent.run(prompt)

        st.write(response)
    else:
        st.warning("Unesi upit prije pokretanja analize.")

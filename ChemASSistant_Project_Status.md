# ChemASSistant --- Project Checkpoint

## Checkpoint_1

**Status:** deterministic cheminformatics core + first working agent
layer\
**Development:** VS Code + Google Drive sync + Google Colab\
**Current development LLM:** Gemini 3.5 Flash Lite via `smolagents`

## Project goal

ChemASSistant is intended to become a **local, agentic cheminformatics
platform**.

> **The LLM should not be where chemistry happens. The LLM should
> orchestrate deterministic, verifiable scientific tools.**

The LLM should understand the user's request, select and call scientific
tools, observe their results, and explain them. Chemistry calculations
and numerical facts should come from RDKit/Python/scientific libraries
rather than LLM memory.

Initial use case: analyze molecular libraries of roughly 10--1000
compounds: validate structures, calculate basic physicochemical
descriptors, apply explicit property rules, find molecular similarity,
and summarize the library. Activity prediction and molecule generation
are intentionally postponed.

Long term: local chat/scientific UI, local LLM, local data/RAG,
modelling/QSAR/DOE, reproducibility/audit trails, and eventually
human-approved iterative/active-learning workflows.

## Architecture

``` text
User / UI
    ↓
LLM agent
    ↓
tool selection and orchestration
    ↓
deterministic scientific tools
    ↓
RDKit / Python / modelling libraries
    ↓
structured results
    ↓
LLM interpretation
```

Principles established so far:

-   Deterministic tools produce scientific facts; the LLM orchestrates
    and interprets.
-   Tools should have narrow, clearly separated responsibilities.
-   Prefer structured outputs over prose.
-   Test every scientific tool as ordinary Python before exposing it to
    an agent.
-   Expected parser failures should be handled cleanly; do not globally
    suppress RDKit errors.
-   Similarity scores are representation/metric dependent, not universal
    percentages of chemical similarity.
-   Property-rule screening is not equivalent to predicting whether a
    molecule is a good drug.
-   Distinguish facts returned by tools from LLM interpretation.
-   Avoid overlapping "do everything" tools when narrow tools give
    clearer routing.

## Current project structure

``` text
ChemASSistant/
├── agent.py
├── pipeline.py
├── tools/
│   ├── __init__.py
│   ├── validation.py
│   ├── descriptors.py
│   ├── rules.py
│   ├── similarity.py
│   └── summary.py
├── data/
└── notebooks/
    └── dev.ipynb
```

`.py` files are the real source; the notebook is the development/test
console.

## Deterministic scientific layer --- v1 complete

### `validation.py`

Implemented `validate_smiles()` and `validate_molecules()`.
Invalid/empty input is handled structurally, valid input returns
canonical SMILES, and expected RDKit parser noise is locally blocked
with `rdBase.BlockLogs()`.

### `descriptors.py`

Implemented `molecular_weight()`, `calculate_descriptors()`, and
`calculate_descriptors_batch()`.

Current descriptors: MW, LogP, TPSA, HBD, HBA, rotatable bonds, heavy
atoms, formal charge. Batch processing does not abort because one
molecule is invalid.

### `rules.py`

Implemented `check_lipinski(descriptors)` with explicit checks for MW \>
500, LogP \> 5, HBD \> 5, HBA \> 10. This is property/rule screening,
not "drug-likeness prediction."

### `similarity.py`

Implemented `calculate_similarity()`,
`calculate_pairwise_similarities()`, and `get_top_similar_pairs()`. Uses
the newer RDKit Morgan fingerprint generator API and Tanimoto
similarity. Each unique pair is considered once.

### `summary.py`

Implemented `summarize_validation()` and `summarize_descriptors()` with
basic counts and min/max/mean statistics.

## Pipeline --- v1 complete

`pipeline.py` implements `analyze_library(smiles_list)`:

``` text
validation
    ↓
canonical valid SMILES
    ↓
descriptors
    ↓
property rules
    ↓
similarity
    ↓
summary
```

It has been tested end-to-end with valid and invalid SMILES.

The full pipeline remains useful as deterministic Python, but its agent
wrapper is currently **not exposed to the active agent** because it
overlaps the narrow tools and encouraged inefficient routing.

## Agent layer --- working first version

Current stack:

-   `smolagents 1.26.0`
-   `ToolCallingAgent`
-   Gemini through Google's OpenAI-compatible endpoint
-   current development model: **Gemini 3.5 Flash Lite**

The API key is stored in Colab Secrets as `GEMINI_API_KEY`, not source
code.

### Tools currently exposed

``` text
validate_molecules_tool
calculate_descriptors_tool
check_property_rules_tool
find_similar_molecules_tool
```

`analyze_library_tool` exists but is intentionally excluded.

When the broad tool and narrow tools were simultaneously available,
Flash Lite selected `analyze_library_tool` for a similarity-only
request. Removing it caused correct routing to
`find_similar_molecules_tool(top_n=3)`.

**Lesson:** tool-set design affects agent quality; clear, minimally
overlapping tools can matter as much as model capability.

## Agent behavior demonstrated

The agent has successfully demonstrated:

-   Croatian natural-language requests;
-   autonomous validation-tool selection;
-   autonomous similarity-tool selection;
-   extraction of `top_n` from natural language;
-   multi-tool reasoning/chaining;
-   deterministic RDKit-backed execution;
-   interpretation of structured observations;
-   recovery after an incorrect tool call.

A multi-tool test requesting both similarity and Lipinski screening
caused the agent to call both appropriate tools and ultimately return
the correct results.

## Agent instructions

Behavioral instructions now tell the agent to:

-   use tools for molecule-specific calculations/claims;
-   call only tools needed for the request;
-   faithfully report all requested tool results;
-   avoid adding unsupported molecular names, properties or conclusions;
-   preserve relevant numerical results;
-   retain original molecular inputs across multi-tool workflows;
-   answer in the user's language.

These instructions improved final-answer fidelity.

## Known limitation at Checkpoint_1

Flash Lite occasionally makes a redundant intermediate call. Observed
example:

``` text
find_similar_molecules_tool(correct inputs)
    ↓
check_property_rules_tool(smiles_list=[])
    ↓
check_property_rules_tool(correct inputs)
    ↓
final answer
```

It recovered automatically, but the empty call wastes a step. Decision:
**do not over-engineer the prompt yet**; record this as a known
development-model limitation.

## Colab / Drive notes

For a newly synced `.py` file that `ls` sees but Python cannot import:

``` python
import importlib
importlib.invalidate_caches()
```

For an already imported module that changed:

``` python
import importlib
importlib.reload(module)
```

Do **not** casually unmount Drive. `drive.flush_and_unmount()`
previously broke the Drive transport and notebook state. Explicit
`importlib.reload()` is preferred; IPython autoreload is unsuitable in
the current Colab/Python environment.

## API observations

One `chem_agent.run()` can cause multiple model requests:

``` text
LLM → choose tool
tool → observation
LLM → choose next action
tool → observation
LLM → final answer
```

Gemini 3.6 Flash hit its observed 5 RPM limit during agent testing.
Gemini 3.5 Flash Lite is currently the practical development default. A
local rate limiter was discussed but deliberately not added yet.

## Planned interface

Leading option: **Streamlit**.

Long-term UI concept: chat, molecular file upload, validation summaries,
descriptor/result tables, plots/2D structures, and a scientific
dashboard.

A public demo can use Streamlit Community Cloud + a cloud LLM with
non-sensitive demo data. The eventual secure version should be local:

``` text
local UI
    ↓
local agent
    ↓
Ollama / local LLM
    ↓
local RDKit/scientific tools
    ↓
local data
```

## Roadmap

``` text
Checkpoint_1
Deterministic tools + working multi-tool agent
        ↓
CSV / SDF input and real molecular datasets
        ↓
simple Streamlit interface
        ↓
local LLM backend via Ollama
        ↓
local RAG
        ↓
modelling / QSAR / DOE
        ↓
reproducibility and audit trail
        ↓
human-approved iterative / active-learning loop
        ↓
specialized agents only if justified
```

LoRA/QLoRA is deliberately postponed until there is evidence of
systematic behavioral gaps plus a real dataset/evaluation set. Chemical
knowledge should primarily enter through RAG and deterministic
scientific tools.

## Checkpoint_1

At **Checkpoint_1**, ChemASSistant can perform:

``` text
natural-language request
        ↓
Gemini Flash Lite
        ↓
smolagents ToolCallingAgent
        ↓
autonomous tool selection
        ↓
deterministic RDKit/Python calculation
        ↓
structured observation
        ↓
LLM user-facing explanation
```

The chemistry layer is deterministic and independently testable. The LLM
layer has demonstrated both single-tool routing and multi-tool chaining.

**Next backend milestone:** CSV/SDF ingestion and real datasets.

**Immediate optional task before that:** build a minimal, easily
shareable Streamlit interface around the current working agent.

Any Streamlit work completed after this point should be **appended below
as the next checkpoint/update**, preserving this Checkpoint_1 as project
history.

---

# ChemASSistant --- Checkpoint_1.1

## Shareable Streamlit Demo

After Checkpoint_1, a minimal web interface was added using **Streamlit
1.64.0**.

The goal of this phase was to create a simple, shareable, and easily
removable UI layer without restructuring the existing ChemASSistant
backend.

### Streamlit UI

A new file was added:

`streamlit_app.py`

The interface currently includes:

-   the ChemASSistant title and a short description;
-   a large `st.text_area` for entering requests;
-   a placeholder containing an example query with SMILES strings;
-   an analysis button;
-   display of the agent's response;
-   `layout="wide"` for a more spacious interface.

`st.chat_input()` was tested but abandoned because Streamlit anchors it
to the bottom of the page in the main app layout. For the current
scientific-tool style interface, a regular text area was preferred.

### GitHub

A GitHub repository named `ChemASSistant` was created for the demo
deployment.

Only the files required for the demo were added:

``` text
streamlit_app.py
agent.py
pipeline.py
requirements.txt

tools/
    __init__.py
    validation.py
    descriptors.py
    rules.py
    similarity.py
    summary.py
```

The development notebook, local data, and API keys were not added to the
repository.

### Dependencies

A `requirements.txt` file was added containing:

``` text
streamlit
rdkit
smolagents[openai]
```

This allows Streamlit Community Cloud to install the dependencies
required by the application.

### Streamlit Community Cloud

The GitHub repository was connected to **Streamlit Community Cloud**.

The deployment is working and the application has a public
`*.streamlit.app` URL that can be shared with other users.

### Gemini API Secret

The Gemini API key is **not stored in the GitHub repository**.

It is stored in **Streamlit Secrets** as:

``` toml
GEMINI_API_KEY = "..."
```

The application accesses it using:

``` python
st.secrets["GEMINI_API_KEY"]
```

### Agent Integration

`streamlit_app.py` instantiates the current development model:

`gemini-3.5-flash-lite`

and creates a `ToolCallingAgent` with the four currently active tools:

``` text
validate_molecules_tool
calculate_descriptors_tool
check_property_rules_tool
find_similar_molecules_tool
```

`analyze_library_tool` remains excluded from the active agent because
earlier testing showed that its broad functionality overlaps the
narrower tools and can lead to inefficient tool routing.

The agent uses the behavioral instructions developed during
Checkpoint_1.

### Current End-to-End System

The complete demo path is now:

``` text
user's browser
        ↓
Streamlit Community Cloud
        ↓
streamlit_app.py
        ↓
Gemini 3.5 Flash Lite
        ↓
smolagents ToolCallingAgent
        ↓
selected ChemASSistant tool
        ↓
RDKit / deterministic Python
        ↓
structured result
        ↓
LLM interpretation
        ↓
response in the web interface
```

ChemASSistant is therefore now a **minimal, publicly shareable agentic
cheminformatics demo**.

### Important Limitations of the Current Demo

This is **not** the secure/local deployment planned for the final
system.

User requests pass through cloud infrastructure, and Gemini requests
consume the API quota associated with the owner's API key. The current
public demo should therefore be used only with **non-sensitive data**.

A visible UI warning should eventually be added, for example:

> Demo version --- do not enter confidential or sensitive data.

### Next Planned Step

The next backend milestone remains:

**CSV/SDF ingestion and work with real molecular datasets.**

After that, the Streamlit interface can be extended with file upload,
tabular results, molecular structures, plots, and other
scientific-dashboard elements.

------------------------------------------------------------------------

**Status at the end of Checkpoint_1.1:** The minimal public Streamlit
demo is operational and connected to the real ChemASSistant agent and
deterministic RDKit tools.

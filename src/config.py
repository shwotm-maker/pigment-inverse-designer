"""Central configuration for Pigment Inverse Designer.

All tunable settings live here so that the rest of the code base only needs
to import from a single, well-documented location (see README section 13).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PROCESSED_CSV: Path = PROCESSED_DIR / "chromophores_clean.csv"
MODEL_PATH: Path = MODELS_DIR / "absorption_model.joblib"
SAMPLE_CSV: Path = RAW_DIR / "sample_chromophores.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Figshare dataset ("DB for chromophore")
# ---------------------------------------------------------------------------
FIGSHARE_ARTICLE_ID: int = 12045567
FIGSHARE_API_URL: str = "https://api.figshare.com/v2/articles/{article_id}"

# ---------------------------------------------------------------------------
# Column auto-mapping.
# For each canonical target we list lowercase substrings; the first raw column
# that contains one of them (and is not already claimed) wins. Order matters.
# ---------------------------------------------------------------------------
COLUMN_MAPPING_RULES: dict[str, list[str]] = {
    "chromophore_smiles": ["chromophore", "dye smiles", "molecule smiles", "canonical_smiles"],
    "solvent_smiles": ["solvent smiles", "solvent"],
    "solvent_name": ["solvent name"],
    "absorption_max": ["absorption max", "absorption maximum", "abs max", "lambda_max_abs", "absorption"],
    "emission_max": ["emission max", "emission maximum", "emi max", "lambda_max_em", "emission"],
    "extinction": ["log(e", "extinction", "epsilon", "molar absorptivity"],
    "quantum_yield": ["quantum yield", "quantum_yield"],
}

# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------
MORGAN_RADIUS: int = 2
MORGAN_NBITS: int = 2048
USE_SOLVENT_FEATURES: bool = True

# RDKit descriptors computed for every molecule (order is significant).
DESCRIPTOR_NAMES: list[str] = [
    "mol_weight",
    "logp",
    "tpsa",
    "aromatic_rings",
    "rotatable_bonds",
    "num_h_acceptors",
    "num_h_donors",
]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_TYPE: str = "extra_trees"  # "extra_trees" | "random_forest"
N_ESTIMATORS: int = 300
MAX_DEPTH: int | None = None
N_JOBS: int = -1

# Reference std (nm) used to normalise the uncertainty indicator to ~[0, 1].
UNCERTAINTY_REFERENCE_NM: float = 40.0

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
SPLIT_METHOD_DEFAULT: str = "scaffold"  # "scaffold" | "random"
TEST_FRACTION: float = 0.15
VAL_FRACTION: float = 0.15

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
TOP_PARENTS_FOR_BRICS: int = 60          # how many close parents to fragment
MAX_VIRTUAL_CANDIDATES: int = 200        # hard cap on generated molecules
BRICS_BUILD_LIMIT: int = 5000            # raw BRICS build attempts before filtering
DEFAULT_TARGET_TOLERANCE_NM: float = 30.0
DEFAULT_N_RESULTS: int = 20

# ---------------------------------------------------------------------------
# Default molecular constraints (used to seed the UI)
# ---------------------------------------------------------------------------
DEFAULT_CONSTRAINTS: dict[str, object] = {
    "min_mol_weight": 100.0,
    "max_mol_weight": 900.0,
    "allowed_elements": ["C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si"],
    "excluded_elements": [],
    "max_logp": 8.0,
    "max_formal_charge": 2,
    "max_rotatable_bonds": 15,
    "min_similarity": 0.0,
    "max_similarity": 1.0,
}

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 100). See scoring.py / README section 8.
# ---------------------------------------------------------------------------
SCORE_WEIGHTS: dict[str, float] = {
    "wavelength": 40.0,
    "uncertainty": 20.0,
    "similarity": 20.0,
    "constraints": 15.0,
    "validity": 5.0,
}
# Similarity band that is rewarded most highly (novel but not detached).
SIMILARITY_SWEET_SPOT: tuple[float, float] = (0.35, 0.75)

# ---------------------------------------------------------------------------
# Optional AI (Claude) language layer. See src/ai_assistant.py.
# Numeric λmax prediction always stays with the local ML model; the LLM only
# does natural-language parsing, result interpretation and qualitative comments.
# The app runs fully without this (no key -> AI features are hidden/disabled).
# ---------------------------------------------------------------------------
# Provider registry. Anthropic uses its own SDK; everything else is reached
# through the OpenAI-compatible Chat Completions API (one `openai` client,
# different base_url). Free-tier providers need only a signup, no card.
# Model names occasionally change — override per-provider in the UI if needed.
AI_PROVIDERS: dict[str, dict] = {
    "Google Gemini (무료)": {
        "sdk": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "key_env": "GEMINI_API_KEY",
        "signup": "https://aistudio.google.com/apikey",
        "note": "카드 없이 무기한 무료. Flash 하루 ~1,500회.",
    },
    "Groq (무료·초고속)": {
        "sdk": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "signup": "https://console.groq.com/keys",
        "note": "오픈모델 초고속. 분당 ~30회.",
    },
    "OpenRouter (무료)": {
        "sdk": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
        "key_env": "OPENROUTER_API_KEY",
        "signup": "https://openrouter.ai/keys",
        "note": "무료 모델 다수. 무료계정 하루 ~50회.",
    },
    "Cerebras (무료)": {
        "sdk": "openai",
        "base_url": "https://api.cerebras.ai/v1",
        "default_model": "llama-3.3-70b",
        "key_env": "CEREBRAS_API_KEY",
        "signup": "https://cloud.cerebras.ai",
        "note": "카드 없이 무료 티어.",
    },
    "Mistral (무료)": {
        "sdk": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
        "key_env": "MISTRAL_API_KEY",
        "signup": "https://console.mistral.ai/api-keys",
        "note": "무료 티어 제공.",
    },
    "NVIDIA NIM (무료)": {
        "sdk": "openai",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "meta/llama-3.1-70b-instruct",
        "key_env": "NVIDIA_API_KEY",
        "signup": "https://build.nvidia.com",
        "note": "카드 없이 무료 티어.",
    },
    "Claude (Anthropic·유료)": {
        "sdk": "anthropic",
        "base_url": None,
        "default_model": "claude-opus-4-8",
        "key_env": "ANTHROPIC_API_KEY",
        "signup": "https://console.anthropic.com/settings/keys",
        "note": "최고 품질, 유료.",
    },
}
AI_DEFAULT_PROVIDER: str = "Google Gemini (무료)"

# ---------------------------------------------------------------------------
# Common solvents offered in the UI (name -> SMILES, "" means solid/unknown)
# ---------------------------------------------------------------------------
COMMON_SOLVENTS: dict[str, str] = {
    "None / solid / unknown": "",
    "Water": "O",
    "Ethanol": "CCO",
    "Methanol": "CO",
    "Acetonitrile": "CC#N",
    "Dichloromethane": "ClCCl",
    "Chloroform": "ClC(Cl)Cl",
    "Toluene": "Cc1ccccc1",
    "Tetrahydrofuran": "C1CCOC1",
    "Dimethyl sulfoxide": "CS(=O)C",
    "N,N-Dimethylformamide": "CN(C)C=O",
    "Cyclohexane": "C1CCCCC1",
    "Acetone": "CC(=O)C",
}

# ---------------------------------------------------------------------------
# Mandatory disclaimer shown on every candidate screen (README section 15).
# ---------------------------------------------------------------------------
CANDIDATE_DISCLAIMER: str = (
    "본 결과는 공개 데이터와 머신러닝 모델에 기반한 가상 스크리닝 결과입니다. "
    "실제 색상, 안전성, 합성 가능성, 물성 및 산업적 사용 가능성은 실험과 전문가 "
    "검토를 통해 별도로 확인해야 합니다."
)

"""Pigment Inverse Designer - Streamlit application.

Research-grade proof-of-concept. Run with:  streamlit run app.py

The app never crashes on missing data or an untrained model: it falls back to a
built-in sample dataset and trains a quick baseline on first launch.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import ai_assistant, config
from src.candidate_search import add_model_predictions, search_existing
from src.data_loader import ensure_sample_dataset, load_and_map, read_uploaded_csv
from src.descriptors import canonical_smiles
from src.model import ModelBundle, load_bundle, save_bundle, train_model
from src.molecule_generator import Constraints, generate_candidates
from src.preprocessing import clean_dataframe, split_dataset
from src.utils import get_logger, set_global_seed
from src.visualization import (
    absorption_histogram,
    candidates_images_zip,
    error_distribution,
    hex_to_absorption_nm,
    mol_image,
    mol_png_bytes,
    parity_plot,
    wavelength_to_rgb,
)

logger = get_logger("app")
set_global_seed()

st.set_page_config(page_title="Pigment Inverse Designer", page_icon="🎨", layout="wide")

EXPORT_COLUMNS = [
    "rank",
    "candidate_type",
    "canonical_smiles",
    "predicted_absorption_nm",
    "experimental_absorption_nm",
    "target_difference_nm",
    "uncertainty",
    "max_training_similarity",
    "molecular_weight",
    "logp",
    "tpsa",
    "aromatic_ring_count",
    "score",
    "warning",
]


# ---------------------------------------------------------------------------
# Cached data / model pipeline
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and model ...")
def get_pipeline(split_method: str) -> dict:
    """Load & clean data, load-or-train the model, attach predictions.

    Cached as a resource so it is built once per split method per session.
    """
    ensure_sample_dataset()
    mapped, mapping = load_and_map()
    clean, report = clean_dataframe(mapped)

    bundle = load_bundle()
    if bundle is None and not clean.empty:
        train_df, val_df, test_df = split_dataset(clean, method=split_method)
        bundle = train_model(train_df, val_df, test_df, split_method=split_method)
        save_bundle(bundle)

    clean_pred = add_model_predictions(clean, bundle) if bundle is not None else clean
    return {
        "mapped": mapped,
        "mapping": mapping,
        "clean": clean_pred,
        "report": report.as_dict(),
        "bundle": bundle,
    }


def retrain_pipeline(clean: pd.DataFrame, split_method: str) -> ModelBundle:
    """Force a retrain (used by the Model Quality tab)."""
    train_df, val_df, test_df = split_dataset(clean, method=split_method)
    bundle = train_model(train_df, val_df, test_df, split_method=split_method)
    save_bundle(bundle)
    return bundle


def disclaimer_banner() -> None:
    st.warning(config.CANDIDATE_DISCLAIMER)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🎨 Pigment Inverse Designer")
st.sidebar.caption("연구용 개념검증 (PoC) · v" + "0.1.0")
split_method = st.sidebar.selectbox(
    "데이터 분할 방식",
    options=["scaffold", "random"],
    index=0 if config.SPLIT_METHOD_DEFAULT == "scaffold" else 1,
    help="scaffold: 유사 골격이 train/test에 섞이지 않음(권장). "
    "random: 구현은 단순하나 유사 구조가 학습·테스트에 동시에 포함될 수 있음.",
)
if split_method == "random":
    st.sidebar.info("⚠️ 랜덤 분할은 유사 구조가 학습·테스트에 동시에 포함될 수 있어 "
                    "성능이 낙관적으로 보일 수 있습니다.")

st.sidebar.divider()
st.sidebar.subheader("🤖 AI 도우미 (선택)")
st.sidebar.caption("자연어 입력·결과 해석·후보 코멘트를 제공합니다. "
                   "예측(λmax)은 항상 로컬 ML이 담당합니다. 무료 공급자도 사용 가능.")
_provider_names = list(config.AI_PROVIDERS.keys())
_ai_provider = st.sidebar.selectbox(
    "AI 공급자", options=_provider_names,
    index=_provider_names.index(config.AI_DEFAULT_PROVIDER),
    help="Gemini·Groq·OpenRouter·Cerebras·Mistral·NVIDIA는 카드 없이 무료 티어. "
         "Claude는 유료·최고품질.",
)
_spec = config.AI_PROVIDERS[_ai_provider]
_env_key = os.environ.get(_spec.get("key_env", ""), "")
st.sidebar.caption(f"ℹ️ {_spec.get('note','')}  |  [키 발급]({_spec['signup']})")
_ai_key_input = st.sidebar.text_input(
    f"{_spec['key_env']}", value="", type="password",
    placeholder=("환경변수에서 감지됨" if _env_key else "여기에 붙여넣기 (선택)"),
    help="입력한 키는 이 세션에서만 사용됩니다. "
         f"환경변수 {_spec['key_env']}가 있으면 비워둬도 됩니다.",
)
_ai_model_override = st.sidebar.text_input(
    "모델명(선택 · 비우면 기본값)", value="",
    placeholder=_spec["default_model"],
    help="공급자 모델명이 바뀌었을 때만 직접 입력하세요.",
)
AI_SETTINGS = ai_assistant.AISettings(
    provider=_ai_provider,
    api_key=(_ai_key_input.strip() or _env_key or None),
    model=(_ai_model_override.strip() or None),
)
AI_ON = ai_assistant.is_available(AI_SETTINGS)
st.sidebar.caption(("✅ AI 사용 가능 (" + _ai_provider + ")") if AI_ON
                   else "⚪ AI 미사용 (키 없음) — 기본 기능은 정상 동작")

pipe = get_pipeline(split_method)
clean_df: pd.DataFrame = pipe["clean"]
bundle: ModelBundle | None = pipe["bundle"]

st.sidebar.metric("유효 발색단 행 수", len(clean_df))
if bundle is not None:
    st.sidebar.metric("Test MAE (nm)", f"{bundle.metrics.get('test', {}).get('mae', float('nan')):.1f}")

tabs = st.tabs(["Overview", "Candidate Search", "Candidate Results", "Model Quality", "Dataset"])


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Overview")
    st.markdown(
        """
**Pigment Inverse Designer** 는 원하는 색상 또는 목표 최대흡수파장(λmax)을 입력하면
공개 발색단 데이터셋에서 가까운 기존 후보를 찾고, RDKit **BRICS** 재조합으로
가상 신규 후보를 생성·평가하는 로컬 웹 애플리케이션입니다.

**사용 방법**
1. **Candidate Search** 탭에서 목표 색상 또는 목표 λmax(nm), 용매, 분자 제약조건을 설정합니다.
2. *기존 후보 검색* 또는 *가상 후보 생성* 버튼을 누릅니다.
3. **Candidate Results** 탭에서 후보 카드와 종합점수를 확인하고 CSV/이미지로 내보냅니다.
4. **Model Quality** 탭에서 모델 성능을, **Dataset** 탭에서 데이터 품질을 확인합니다.
        """
    )
    disclaimer_banner()
    st.subheader("데이터셋 개요")
    st.markdown(
        f"""
- 출처: Figshare *DB for chromophore* (article ID **{config.FIGSHARE_ARTICLE_ID}**),
  약 7,000개 발색단 / 20,000+ 발색단·용매 조합.
- 다운로드 실패 시 내장 샘플 데이터셋(수십 개 대표 발색단)으로 자동 대체됩니다.
- 현재 로드된 유효 행 수: **{len(clean_df)}**
        """
    )
    st.error(
        "이 프로그램은 실제 합성 성공, 색상 정확도, 독성/안전성, 특허 신규성, "
        "산업 생산성을 **보증하지 않으며**, 구체적 합성 절차나 반응조건을 생성하지 않습니다."
    )


# ---------------------------------------------------------------------------
# Candidate Search
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Candidate Search")
    disclaimer_banner()

    with st.expander("🤖 자연어로 목표 입력 (AI, 선택)", expanded=False):
        if not AI_ON:
            st.info("사이드바에 ANTHROPIC_API_KEY를 입력하면 활성화됩니다. "
                    "키가 없어도 아래 수동 입력으로 모든 기능을 쓸 수 있습니다.")
        nl_text = st.text_input(
            "예: '청록색 계열, 물에 잘 녹고 분자량 500 이하로 찾아줘'",
            key="nl_target", disabled=not AI_ON,
        )
        if st.button("AI로 조건 해석", disabled=not AI_ON) and nl_text:
            with st.spinner("Claude가 조건을 해석 중..."):
                res = ai_assistant.parse_natural_language_target(nl_text, AI_SETTINGS)
            if res.ok:
                d = res.content
                st.session_state["nl_parsed"] = d
                if d.get("target_absorption_nm"):
                    st.session_state["target_nm_override"] = float(d["target_absorption_nm"])
                st.success("해석 완료 — 목표 λmax는 자동 반영, 나머지는 제안값입니다(아래에서 조정).")
                st.caption("근거: " + str(d.get("reasoning", "")))
                st.json({k: v for k, v in d.items() if k != "reasoning"})
                st.rerun()
            else:
                st.error(res.error)

    col_target, col_color = st.columns(2)
    with col_target:
        st.subheader("목표 최대흡수파장 (권장)")
        target_nm = st.number_input(
            "목표 λmax (nm)", min_value=200.0, max_value=1200.0,
            value=500.0, step=5.0,
            help="과학적 판단이 필요한 경우 색상 선택기 대신 nm 직접 입력을 권장합니다.",
        )
        st.caption("흡수 λmax에 해당하는 근사 색상 미리보기:")
        st.color_picker("근사 흡수색(참고용)", value="#%02x%02x%02x" % wavelength_to_rgb(target_nm),
                        disabled=True, key="abs_preview")

    with col_color:
        st.subheader("목표 색상으로 선택 (데모)")
        picked = st.color_picker("원하는 외관 색상", value="#1f77b4")
        demo_nm = hex_to_absorption_nm(picked)
        st.caption(f"보색 기반 근사 흡수 λmax ≈ **{demo_nm:.0f} nm** (데모용 근사치)")
        if st.button("이 색상의 근사 λmax를 목표로 사용"):
            st.session_state["target_nm_override"] = float(demo_nm)
            st.rerun()
        st.info(
            "색상과 최대흡수파장은 일대일 대응이 아닙니다. 실제 색은 전체 흡수 스펙트럼, "
            "결정형, 입자크기, 농도, 분산상태, 측정조건에 따라 달라집니다. "
            "색상→파장 변환은 단순 데모용 근사치입니다."
        )

    if "target_nm_override" in st.session_state:
        target_nm = st.session_state.pop("target_nm_override")
        st.success(f"목표 λmax를 {target_nm:.0f} nm 로 설정했습니다.")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        solvent_name = st.selectbox("용매 / 상태", options=list(config.COMMON_SOLVENTS.keys()))
        solvent_smiles = config.COMMON_SOLVENTS[solvent_name] or None
    with c2:
        tolerance_nm = st.number_input("목표파장 허용오차 (± nm)", 5.0, 200.0,
                                       config.DEFAULT_TARGET_TOLERANCE_NM, 5.0)
    with c3:
        n_results = st.number_input("출력 후보 개수", 1, 100, config.DEFAULT_N_RESULTS, 1)

    with st.expander("분자 제약조건 (가상 후보 생성용)", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            min_mw = st.number_input("최소 분자량", 0.0, 2000.0,
                                     float(config.DEFAULT_CONSTRAINTS["min_mol_weight"]), 10.0)
            max_mw = st.number_input("최대 분자량", 0.0, 3000.0,
                                     float(config.DEFAULT_CONSTRAINTS["max_mol_weight"]), 10.0)
            max_logp = st.number_input("최대 LogP", -5.0, 15.0,
                                       float(config.DEFAULT_CONSTRAINTS["max_logp"]), 0.5)
        with cc2:
            max_charge = st.number_input("최대 |formal charge|", 0, 6,
                                         int(config.DEFAULT_CONSTRAINTS["max_formal_charge"]), 1)
            max_rot = st.number_input("최대 회전가능결합 수", 0, 40,
                                      int(config.DEFAULT_CONSTRAINTS["max_rotatable_bonds"]), 1)
        with cc3:
            min_sim = st.slider("기존 데이터셋과의 최소 유사도", 0.0, 1.0,
                                float(config.DEFAULT_CONSTRAINTS["min_similarity"]), 0.05)
            max_sim = st.slider("기존 데이터셋과의 최대 유사도", 0.0, 1.0,
                                float(config.DEFAULT_CONSTRAINTS["max_similarity"]), 0.05)
        allowed = st.multiselect("허용 원소",
                                 options=["C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"],
                                 default=list(config.DEFAULT_CONSTRAINTS["allowed_elements"]))
        excluded = st.multiselect("제외 원소",
                                  options=["F", "Cl", "Br", "I", "P", "B", "Si", "Se", "As", "Hg"],
                                  default=list(config.DEFAULT_CONSTRAINTS["excluded_elements"]))
        n_parents = st.slider("BRICS 부모 분자 수", 30, 100, config.TOP_PARENTS_FOR_BRICS, 5)
        max_cand = st.slider("최대 가상 후보 수(부하 제한)", 20, 500,
                             config.MAX_VIRTUAL_CANDIDATES, 20)

    constraints = Constraints(
        min_mol_weight=min_mw, max_mol_weight=max_mw, allowed_elements=allowed,
        excluded_elements=excluded, max_logp=max_logp, max_formal_charge=max_charge,
        max_rotatable_bonds=max_rot, min_similarity=min_sim, max_similarity=max_sim,
    )

    st.session_state["search_ctx"] = dict(
        target_nm=float(target_nm), solvent_smiles=solvent_smiles,
        tolerance_nm=float(tolerance_nm), n_results=int(n_results),
    )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("🔎 기존 후보 검색", use_container_width=True):
            if bundle is None:
                st.error("모델이 준비되지 않았습니다. Dataset 탭에서 데이터를 확인하세요.")
            else:
                exp = search_existing(clean_df, target_nm, by="experimental",
                                      tolerance_nm=tolerance_nm, n_results=n_results,
                                      solvent_filter=None)
                pred = search_existing(clean_df, target_nm, by="predicted",
                                       tolerance_nm=tolerance_nm, n_results=n_results,
                                       solvent_filter=None)
                st.session_state["existing_exp"] = exp
                st.session_state["existing_pred"] = pred
                st.success(f"실험값 기준 {len(exp)}개, 예측값 기준 {len(pred)}개 후보를 찾았습니다. "
                           "→ Candidate Results 탭 확인")

    with b2:
        if st.button("🧪 가상 후보 생성 (BRICS)", use_container_width=True):
            if bundle is None:
                st.error("모델이 준비되지 않았습니다.")
            else:
                prog = st.progress(0.0, text="시작 중 ...")

                def _cb(frac: float, msg: str) -> None:
                    prog.progress(frac, text=msg)

                virt = generate_candidates(
                    clean_df, bundle, target_nm, solvent_smiles,
                    constraints=constraints, tolerance_nm=tolerance_nm,
                    n_parents=n_parents, max_candidates=max_cand,
                    n_results=n_results, progress_cb=_cb,
                )
                prog.empty()
                st.session_state["virtual"] = virt
                if virt.empty:
                    st.warning("조건을 만족하는 가상 후보가 없습니다. 제약조건을 완화해 보세요.")
                else:
                    st.success(f"가상 후보 {len(virt)}개 생성 완료. → Candidate Results 탭 확인")


# ---------------------------------------------------------------------------
# Candidate Results
# ---------------------------------------------------------------------------
def render_cards(df: pd.DataFrame, ctx: dict) -> None:
    """Render candidate cards + export buttons for a result frame."""
    if df is None or df.empty:
        st.info("표시할 후보가 없습니다.")
        return

    df = df.reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    export = build_export_frame(df)

    # --- AI interpretation report (optional) ---
    if AI_ON:
        if st.button("🤖 AI 해석 리포트 생성", key=f"aireport_{ctx.get('kind','x')}"):
            records = export.drop(columns=["warning"]).head(15).to_dict(orient="records")
            with st.spinner("Claude가 후보를 해석하는 중..."):
                res = ai_assistant.interpret_results(
                    float(ctx.get("target_nm", 0.0)), records,
                    ctx.get("kind", "후보"), AI_SETTINGS,
                )
            if res.ok:
                st.session_state[f"aireport_out_{ctx.get('kind','x')}"] = res.content
            else:
                st.error(res.error)
        report = st.session_state.get(f"aireport_out_{ctx.get('kind','x')}")
        if report:
            with st.container(border=True):
                st.markdown("#### 🤖 AI 해석 리포트")
                st.markdown(report)
                st.caption("⚠️ AI 요약은 참고용입니다. " + config.CANDIDATE_DISCLAIMER)

    for _, row in df.iterrows():
        with st.container(border=True):
            left, right = st.columns([1, 2])
            with left:
                img = mol_image(row["canonical_smiles"])
                if img is not None:
                    st.image(img, caption=f"#{int(row['rank'])}")
                else:
                    st.write("(구조 렌더 실패)")
            with right:
                ctype = row.get("candidate_type", "Existing")
                st.markdown(f"**#{int(row['rank'])} · {ctype}**")
                st.code(row["canonical_smiles"], language=None)
                m1, m2, m3 = st.columns(3)
                pred = row.get("predicted_absorption_nm")
                exp = row.get("experimental_absorption_nm", row.get("absorption_max"))
                diff = row.get("target_difference_nm")
                m1.metric("예측 λmax (nm)", "-" if pd.isna(pred) else f"{pred:.0f}")
                m1.metric("실험 λmax (nm)", "-" if pd.isna(exp) else f"{exp:.0f}")
                m2.metric("목표차 (nm)", "-" if pd.isna(diff) else f"{diff:+.0f}")
                unc = row.get("uncertainty")
                m2.metric("불확실성(±nm, 참고)", "-" if pd.isna(unc) else f"{unc:.0f}")
                sim = row.get("max_training_similarity")
                m3.metric("최대 Tanimoto", "-" if pd.isna(sim) else f"{sim:.2f}")
                score = row.get("score")
                m3.metric("종합점수", "-" if pd.isna(score) else f"{score:.0f}/100")
                st.caption(
                    f"MW {row.get('molecular_weight', row.get('mol_weight', float('nan'))):.0f} · "
                    f"LogP {row.get('logp', float('nan')):.1f} · "
                    f"방향족고리 {int(row.get('aromatic_ring_count', row.get('aromatic_rings', 0)))} · "
                    f"출처 {row.get('source', ctype)}"
                )
                if not pd.isna(sim):
                    if sim >= config.SIMILARITY_SWEET_SPOT[1]:
                        st.caption("↳ 유사도 높음: 기존 물질에 가까운 후보(신규성 낮음).")
                    elif sim <= config.SIMILARITY_SWEET_SPOT[0]:
                        st.caption("↳ 유사도 낮음: 신규성 높으나 예측 신뢰도가 낮을 수 있음.")
                st.caption("⚠️ " + config.CANDIDATE_DISCLAIMER)
                if AI_ON:
                    ckey = f"cmt_{ctx.get('kind','x')}_{int(row['rank'])}"
                    if st.button("🤖 AI 코멘트", key=f"btn_{ckey}"):
                        rec = {
                            "smiles": row["canonical_smiles"],
                            "predicted_nm": row.get("predicted_absorption_nm"),
                            "experimental_nm": row.get("experimental_absorption_nm",
                                                       row.get("absorption_max")),
                            "target_difference_nm": row.get("target_difference_nm"),
                            "uncertainty": row.get("uncertainty"),
                            "max_training_similarity": row.get("max_training_similarity"),
                            "mol_weight": row.get("molecular_weight", row.get("mol_weight")),
                            "logp": row.get("logp"),
                            "aromatic_rings": row.get("aromatic_ring_count",
                                                      row.get("aromatic_rings")),
                        }
                        with st.spinner("코멘트 생성 중..."):
                            res = ai_assistant.comment_on_candidate(
                                rec, float(ctx.get("target_nm", 0.0)), AI_SETTINGS)
                        st.session_state[ckey] = res.content if res.ok else f"⚠️ {res.error}"
                    if st.session_state.get(ckey):
                        st.info(st.session_state[ckey])

    # --- exports (export frame already built above) ---
    st.download_button(
        "⬇️ 결과 CSV 다운로드",
        data=export.to_csv(index=False).encode("utf-8-sig"),
        file_name="candidates.csv",
        mime="text/csv",
    )
    zip_bytes = candidates_images_zip(df["canonical_smiles"].tolist())
    st.download_button(
        "⬇️ 구조 이미지 ZIP 다운로드", data=zip_bytes,
        file_name="candidate_structures.zip", mime="application/zip",
    )


def build_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise any result frame to the fixed export schema."""
    out = pd.DataFrame()
    out["rank"] = df.get("rank", np.arange(1, len(df) + 1))
    out["candidate_type"] = df.get("candidate_type", "Existing")
    out["canonical_smiles"] = df.get("canonical_smiles")
    out["predicted_absorption_nm"] = df.get("predicted_absorption_nm")
    out["experimental_absorption_nm"] = df.get("experimental_absorption_nm",
                                               df.get("absorption_max"))
    out["target_difference_nm"] = df.get("target_difference_nm")
    out["uncertainty"] = df.get("uncertainty")
    out["max_training_similarity"] = df.get("max_training_similarity")
    out["molecular_weight"] = df.get("molecular_weight", df.get("mol_weight"))
    out["logp"] = df.get("logp")
    out["tpsa"] = df.get("tpsa")
    out["aromatic_ring_count"] = df.get("aromatic_ring_count", df.get("aromatic_rings"))
    out["score"] = df.get("score")
    out["warning"] = config.CANDIDATE_DISCLAIMER
    return out[EXPORT_COLUMNS]


with tabs[2]:
    st.header("Candidate Results")
    disclaimer_banner()
    ctx = st.session_state.get("search_ctx", {})

    sub = st.radio("후보 종류", ["가상 신규 후보", "기존 후보(실험값)", "기존 후보(예측값)"],
                   horizontal=True)
    ctx = {**ctx, "kind": sub}
    if sub == "가상 신규 후보":
        render_cards(st.session_state.get("virtual"), ctx)
    elif sub == "기존 후보(실험값)":
        render_cards(st.session_state.get("existing_exp"), ctx)
    else:
        render_cards(st.session_state.get("existing_pred"), ctx)


# ---------------------------------------------------------------------------
# Model Quality
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Model Quality")
    if bundle is None:
        st.error("학습된 모델이 없습니다.")
    else:
        m = bundle.metrics
        cols = st.columns(3)
        for i, part in enumerate(["train", "val", "test"]):
            mm = m.get(part, {})
            with cols[i]:
                st.markdown(f"**{part.upper()}** (n={mm.get('n', 0)})")
                st.metric("MAE (nm)", f"{mm.get('mae', float('nan')):.1f}")
                st.metric("RMSE (nm)", f"{mm.get('rmse', float('nan')):.1f}")
                st.metric("R²", f"{mm.get('r2', float('nan')):.3f}")

        st.caption(f"분할 방식: **{bundle.split_method}** · "
                   "불확실성 지표는 트리별 예측 표준편차(모델 내부 분산 기반 참고값이며 "
                   "통계적 신뢰구간이 아님).")

        st.divider()
        # Recompute test predictions for plots/table.
        from src.model import predict_with_uncertainty

        train_df, val_df, test_df = split_dataset(clean_df, method=bundle.split_method)
        preds, unc, valid_idx = predict_with_uncertainty(
            bundle, test_df["canonical_smiles"].tolist(), test_df["solvent_smiles"].tolist())
        y_true = test_df.iloc[valid_idx]["absorption_max"].to_numpy(dtype=float)

        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(parity_plot(y_true, preds))
        with g2:
            st.pyplot(error_distribution(y_true, preds))
        st.pyplot(absorption_histogram(bundle.train_absorption, "학습 데이터 흡수파장 분포"))

        st.subheader("테스트 데이터 예측 테이블")
        tbl = test_df.iloc[valid_idx][["canonical_smiles", "solvent_smiles", "absorption_max"]].copy()
        tbl["predicted_nm"] = np.round(preds, 1)
        tbl["uncertainty_nm"] = np.round(unc, 1)
        tbl["error_nm"] = np.round(preds - y_true, 1)
        st.dataframe(tbl.reset_index(drop=True), use_container_width=True, height=320)

        if st.button("🔁 모델 재학습"):
            get_pipeline.clear()
            st.session_state["_retrain"] = True
            new_bundle = retrain_pipeline(clean_df, split_method)
            st.success(f"재학습 완료. Test MAE={new_bundle.metrics['test']['mae']:.1f} nm")
            st.rerun()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Dataset")
    st.caption(f"컬럼 자동 매핑 결과: {pipe['mapping']}")
    rep = pipe["report"]
    c = st.columns(5)
    c[0].metric("입력 행", rep.get("input_rows", 0))
    c[1].metric("무효 SMILES", rep.get("invalid_smiles", 0))
    c[2].metric("흡수값 결측", rep.get("missing_absorption", 0))
    c[3].metric("중복 제거", rep.get("duplicates_removed", 0))
    c[4].metric("클린 행", rep.get("clean_rows", 0))

    st.subheader("데이터 미리보기")
    st.dataframe(clean_df.head(50), use_container_width=True, height=320)

    st.subheader("흡수파장 분포")
    st.pyplot(absorption_histogram(clean_df["absorption_max"].to_numpy(),
                                   "전체 데이터 흡수파장 분포"))

    st.subheader("CSV 업로드 (사내 실험데이터 추가)")
    st.caption("chromophore SMILES / solvent / absorption max 컬럼이 있으면 자동 매핑됩니다.")
    up = st.file_uploader("CSV 또는 Excel 업로드", type=["csv", "xlsx", "xls"])
    if up is not None:
        try:
            raw = read_uploaded_csv(up)
            from src.data_loader import map_columns

            mapped_up, mapping_up = map_columns(raw)
            mapped_up["source"] = f"upload:{up.name}"
            clean_up, rep_up = clean_dataframe(mapped_up)
            st.success(f"업로드 매핑: {mapping_up}")
            st.write(rep_up.as_dict())
            st.dataframe(clean_up.head(30), use_container_width=True)
            if st.button("업로드 데이터를 raw 폴더에 저장"):
                dest = config.RAW_DIR / f"user_{up.name}"
                raw.to_csv(dest, index=False, encoding="utf-8")
                get_pipeline.clear()
                st.success(f"{dest} 저장됨. 사이드바에서 다시 로드되며, "
                           "재학습하려면 Model Quality 탭의 재학습 버튼을 사용하세요.")
        except Exception as exc:
            st.error(f"업로드 처리 실패: {exc}")

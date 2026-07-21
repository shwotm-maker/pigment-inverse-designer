# Pigment Inverse Designer

원하는 **색상** 또는 목표 **최대흡수파장(λmax)** 을 입력하면, 공개 발색단
데이터셋에서 가까운 기존 후보를 검색하고 RDKit **BRICS** 재조합으로 가상 신규
후보 구조를 생성·평가하는 **로컬** 웹 애플리케이션입니다.

> ⚠️ **연구용 개념검증(PoC) 프로그램입니다.** 실제 합성 가능성, 산업용 안료 성능,
> 색상 정확도, 안전성/독성, 특허 신규성을 **보증하지 않습니다.**

---

## 1. 프로그램의 목적
- 목표 광학특성(λmax) 기반의 **역설계(inverse design)** 워크플로 시연
- 기존 발색단 **탐색** + 가상 신규 후보 **생성/스크리닝** + **종합점수** 산정
- 완전 오프라인·로컬 실행(유료 API·LLM·클라우드 미사용)

## 2. 프로그램의 한계 (매우 중요)
이 앱은 다음을 **하지 않습니다.**
- 실제 합성 성공/절차/반응조건 생성 또는 보증
- 색상 정확도 보증 (색과 λmax는 일대일 대응이 아님)
- 독성·안전성·특허 신규성·산업 생산성 보증

모든 후보 화면에는 다음 문구가 표시됩니다.

> 본 결과는 공개 데이터와 머신러닝 모델에 기반한 가상 스크리닝 결과입니다.
> 실제 색상, 안전성, 합성 가능성, 물성 및 산업적 사용 가능성은 실험과 전문가
> 검토를 통해 별도로 확인해야 합니다.

---

## 3. 설치 방법

### conda 환경 (권장 — RDKit 설치가 가장 안정적)
```bash
conda create -n pigment python=3.11 -y
conda activate pigment
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

### pip 환경
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt   # rdkit는 pip(rdkit) 휠로 설치됩니다
```

---

## 4. 데이터 다운로드 방법
공개 **"DB for chromophore"** 데이터셋 (Figshare article ID **12045567**,
약 7,000개 발색단 / 20,000+ 발색단·용매 조합).

```bash
python -m scripts.download_data
```
Figshare 공개 API로 파일 목록과 실제 download URL을 조회해 `data/raw/` 에 저장합니다.

### 다운로드 실패 시 (수동)
1. 브라우저에서 `https://figshare.com/articles/dataset/_/12045567` 접속
2. 발색단 CSV/Excel 파일 다운로드
3. `data/raw/` 폴더에 복사
4. `python -m scripts.train_model` 재실행

> 다운로드가 안 되어도 **내장 샘플 데이터셋**으로 앱과 모델이 즉시 동작합니다.

### 컬럼 자동 매핑
실제 CSV의 컬럼명을 먼저 검사하여 아래 의미로 자동 매핑합니다
(`src/config.py`의 `COLUMN_MAPPING_RULES`). 특정 컬럼이 없어도 중단되지 않습니다.
- chromophore SMILES / solvent SMILES(또는 solvent name) / absorption max /
  emission max / extinction coefficient / quantum yield

---

## 5. 모델 학습 방법
```bash
python -m scripts.train_model            # scaffold split (기본, 권장)
python -m scripts.train_model --split random
```
- 모델: `ExtraTreesRegressor` (baseline, 재현 가능·빠름)
- 입력: Morgan fingerprint(2048) + RDKit descriptors(+ 용매 descriptor)
- 목표값: maximum absorption wavelength (nm)
- 산출물: `models/absorption_model.joblib`, `data/processed/chromophores_clean.csv`

### 분할 방식
- **scaffold split (권장):** Bemis–Murcko scaffold 기준으로 유사 골격이
  train/test에 동시에 포함되지 않도록 분리 → 일반화 성능을 정직하게 평가.
- **random split:** 구현은 단순하지만 **유사 구조가 학습·테스트에 동시에 포함**될
  수 있어 성능이 낙관적으로 보일 수 있습니다(UI/README에 명시).

---

## 6. 실행 방법

### Streamlit 실행
```bash
streamlit run app.py
```

### Windows 원클릭 실행
`run_windows.bat` 더블클릭 → venv 생성·의존성 설치·데이터 다운로드·모델 학습·앱 실행을
자동 수행합니다.

---

## 7. 화면 구성
- **Overview** — 프로젝트 설명/사용법/경고/데이터 개요
- **Candidate Search** — 목표 색상·λmax·용매·제약조건 입력, 기존/가상 후보 실행
- **Candidate Results** — 후보 카드(구조·SMILES·예측/실험 λmax·목표차·불확실성·
  Tanimoto·descriptor·종합점수·주의사항), CSV/이미지 ZIP 내보내기
- **Model Quality** — MAE/RMSE/R², 실제 대 예측 산점도, 오차 분포, 데이터 분포,
  테스트 예측 테이블, 재학습 버튼
- **Dataset** — 미리보기, 유효/무효 SMILES·결측 통계, 파장 분포, CSV 업로드

---

## 8. 후보 종합점수 (0~100)
`src/scoring.py`의 `compute_score()`가 5개 요소를 가중합합니다
(가중치는 `config.SCORE_WEIGHTS`, 합계 100).

| 요소 | 가중치 | 의미 |
|---|---|---|
| wavelength | 40 | 예측 λmax가 목표에 가까울수록 높음(허용오차의 3배에서 0으로 감쇠) |
| uncertainty | 20 | 트리 분산(±nm)이 작을수록 높음(기준 std에서 0) |
| similarity | 20 | 학습셋과의 최대 Tanimoto가 **신규성 sweet-spot 밴드**(기본 0.35~0.75)일 때 최대 |
| constraints | 15 | 사용자 제약조건 충족 비율 |
| validity | 5 | RDKit sanitize 통과 여부 |

> "합성 가능성 점수"라는 표현은 사용하지 않습니다. 이 점수는 **스크리닝 보조 지표**입니다.

**불확실성 지표**는 트리별 예측값의 표준편차로, **모델 내부 예측 분산에 기반한
참고값**이며 통계적 신뢰구간이 아닙니다.

---

## 9. 색상 인터페이스에 관한 주의
- λmax **직접 입력(nm)** 과 색상 **선택기** 두 방식을 제공합니다.
- **색상과 λmax는 일대일 대응이 아닙니다.** 실제 색은 전체 흡수 스펙트럼, 결정형,
  입자크기, 농도, 분산상태, 측정조건의 영향을 받습니다.
- 색상→파장 변환은 **보색 기반 데모용 근사치**입니다. 과학적 판단이 필요한 화면에서는
  **nm 직접 입력을 권장**합니다.

---

## 10. 후보 결과 해석법
- **목표차(nm):** 예측 λmax − 목표. 0에 가까울수록 목표 부합.
- **불확실성(±nm):** 클수록 모델 예측 신뢰도가 낮음(참고값).
- **최대 Tanimoto:** 높으면 기존 물질에 가까움(신규성 낮음), 낮으면 신규성은
  높으나 예측 신뢰도가 낮을 수 있음.
- **Virtual candidate** 라벨은 BRICS 재조합으로 생성된 **가상** 구조임을 의미하며,
  실제 합성 가능성을 보장하지 않습니다.

## AI 도우미 (선택 · 여러 공급자 지원)
정량 예측(λmax)은 **항상 로컬 ML 모델**이 담당합니다. AI(LLM)는 **말로 하는 부분**만
보조합니다 — LLM은 정량 파장 예측에서 신뢰도가 낮기 때문입니다.

- **자연어 목표 입력**: "청록색, 물에 잘 녹는 걸로" → 목표 nm·용매·제약조건 자동 해석
- **결과 해석 리포트**: 상위 후보를 한국어로 "왜 유망한지·주의점" 요약
- **후보별 AI 코멘트**: 구조 특징·예상 성질 짧은 요약

**공급자 선택 (사이드바)** — 무료 티어로 대체 가능합니다. 성능은 낮아도 이 언어 계층엔 충분합니다.

| 공급자 | 무료 여부 | 키 발급 | 환경변수 |
|---|---|---|---|
| **Google Gemini** (기본값) | 카드 없이 무기한 무료(Flash 하루 ~1,500회) | aistudio.google.com/apikey | `GEMINI_API_KEY` |
| **Groq** | 무료·초고속(분당 ~30회) | console.groq.com/keys | `GROQ_API_KEY` |
| **OpenRouter** | 무료 모델 다수(하루 ~50회) | openrouter.ai/keys | `OPENROUTER_API_KEY` |
| **Cerebras** | 카드 없이 무료 티어 | cloud.cerebras.ai | `CEREBRAS_API_KEY` |
| **Mistral** | 무료 티어 | console.mistral.ai/api-keys | `MISTRAL_API_KEY` |
| **NVIDIA NIM** | 카드 없이 무료 티어 | build.nvidia.com | `NVIDIA_API_KEY` |
| **Claude (Anthropic)** | 유료·최고품질 | console.anthropic.com | `ANTHROPIC_API_KEY` |

> 기술적으로 Gemini/Groq/OpenRouter/Cerebras/Mistral/NVIDIA는 모두 **OpenAI 호환 API**라
> `openai` SDK 하나로 base_url만 바꿔 붙습니다(Claude만 `anthropic` SDK). 공급자 목록·모델명은
> `src/config.py`의 `AI_PROVIDERS`에서 관리합니다.

**켜는 법**
1. `pip install -r requirements.txt` (`openai`, `anthropic` 포함)
2. 사이드바에서 **공급자 선택** → 해당 키를 입력칸에 붙여넣기(세션에서만 사용) 또는 환경변수 설정
   - 예) PowerShell: `\$env:GEMINI_API_KEY = "..."`
3. 모델명이 바뀌었으면 사이드바의 "모델명(선택)"에 직접 입력. 비우면 공급자별 기본값 사용.
4. 키가 없으면 AI 컨트롤은 비활성화되고 **나머지 기능은 그대로 오프라인 동작**합니다.

**한계**: AI 요약은 참고용이며 합성 절차·반응조건을 생성하지 않도록 제약되어 있습니다.
AI 기능은 클라우드 API라 인터넷이 필요합니다. 포터블 ZIP은 ML 기능은 오프라인으로,
AI 기능은 키+인터넷이 있을 때만 동작합니다.

## 11. 산업용 안료에 바로 적용할 수 없는 이유
- 데이터는 주로 **용액상 광학 특성**이며, 안료의 실제 발색은 고체상/결정형/입자/분산에
  크게 좌우됨.
- 견뢰도(내광·내열·내약품성), 독성/규제, 분산성, 원가, 합성 난이도 등은 모델에
  포함되지 않음.
- baseline 모델은 학습 분포 밖(외삽)에서 신뢰도가 급격히 낮아짐.

## 12. 사내 실험데이터 추가 방법
1. **Dataset** 탭 → *CSV 업로드* 로 사내 데이터를 올리면 컬럼이 자동 매핑됩니다.
2. "raw 폴더에 저장" 후 `python -m scripts.train_model` 로 재학습하거나,
   Model Quality 탭의 **재학습** 버튼을 사용합니다.
3. 또는 CSV를 직접 `data/raw/` 에 넣고 재학습해도 됩니다.

---

## 13. 프로젝트 구조
```
pigment-inverse-designer/
├─ app.py                     # Streamlit 앱 (5개 탭)
├─ requirements.txt
├─ README.md
├─ run_windows.bat
├─ data/{raw,processed}/
├─ models/                    # absorption_model.joblib
├─ outputs/
├─ scripts/
│  ├─ download_data.py        # Figshare API 다운로드 + 샘플 보장
│  └─ train_model.py          # load→clean→split→train→save
├─ src/
│  ├─ config.py               # 모든 설정 중앙관리
│  ├─ data_loader.py          # Figshare/CSV 로드 + 컬럼 자동매핑 + 샘플
│  ├─ preprocessing.py        # 검증/정규화/중복/결측/scaffold split
│  ├─ descriptors.py          # descriptors + Morgan fp + 용매 feature
│  ├─ model.py                # ExtraTrees 학습/평가/불확실성/저장
│  ├─ candidate_search.py     # 기존 후보 검색 + Tanimoto
│  ├─ molecule_generator.py   # BRICS 생성/필터/점수
│  ├─ scoring.py              # 0~100 종합점수
│  ├─ visualization.py        # 2D 구조/플롯/색상 근사
│  └─ utils.py                # 로깅/시드/헬퍼
└─ tests/
   ├─ test_preprocessing.py
   ├─ test_descriptors.py
   └─ test_scoring.py
```

## 14. 테스트
```bash
pytest -q
```

## 15. 결과 내보내기
Candidate Results 탭에서 후보를 CSV로 내려받을 수 있습니다. 컬럼:
`rank, candidate_type, canonical_smiles, predicted_absorption_nm,
experimental_absorption_nm, target_difference_nm, uncertainty,
max_training_similarity, molecular_weight, logp, tpsa, aromatic_ring_count,
score, warning`. 구조 이미지는 개별 PNG를 묶은 **ZIP** 으로 저장됩니다.

---

## 16. 다음 버전: Chemprop(D-MPNN)으로 모델 교체
현재 baseline(ExtraTrees + fingerprint)을 그래프 신경망(**Chemprop**, directed
message-passing)으로 교체하는 방법:

1. `pip install chemprop` (PyTorch 필요).
2. 학습 데이터를 Chemprop CSV 포맷으로 저장: `smiles,target` 컬럼
   (`data/processed/chromophores_clean.csv` → `canonical_smiles,absorption_max`).
3. 학습:
   ```bash
   chemprop_train --data_path chemprop.csv --dataset_type regression \
     --smiles_columns canonical_smiles --target_columns absorption_max \
     --split_type scaffold_balanced --save_dir models/chemprop
   ```
4. `src/model.py`에 `ModelBundle` 인터페이스(`predict_with_uncertainty`)를
   유지하는 `ChempropBundle`을 추가하고, 불확실성은 `--ensemble_size` 또는
   MC-dropout으로 산출.
5. 용매 효과는 solvent descriptor를 추가 feature로 넣거나 다중입력 모델로 확장.
6. `app.py`의 `get_pipeline()`이 joblib 대신 Chemprop 체크포인트를 로드하도록
   분기 추가. 나머지 UI/스코어링 코드는 그대로 재사용됩니다.
```
```

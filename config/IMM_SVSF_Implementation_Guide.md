# Phase 5: IMM-SVSF Implementation Guide for Robust Ego-Track Selection

## 1. 개요 (Overview)
본 문서는 기존의 다항식 피팅(Phase 4) 기반 트래커를 **IMM-SVSF (Interacting Multiple Model - Smooth Variable Structure Filter)** 기반으로 업그레이드하기 위한 구현 가이드를 담고 있습니다.  
관련 논문 *"The Interacting Multiple Model Smooth Variable Structure Filter for Trajectory Prediction"*의 핵심 알고리즘을 적용하여, 모델 불확실성과 급격한 기동(분기점)에 강건한 Ego-Track 선택 로직을 구현합니다.

---

## 2. 방법론: 3단계 구현 전략 (Methodology)

### ① 좌표계 정의 (Curvilinear Coordinates)
논문에서 도로 형상을 기준으로 좌표를 단순화한 것처럼, 이미지 스캔 라인을 기준으로 동역학을 정의합니다.
- **종방향 ($s$):** 이미지의 `y` 좌표 (Bottom $\to$ Top 스캔 방향)
- **횡방향 ($l$):** 이미지의 `x` 좌표 (Center Line 또는 예측 경로로부터의 횡방향 편차)
- **상태 변수:** $X = [x, \dot{x}, \ddot{x}]^T$ (횡방향 위치, 기울기, 곡률)

### ② 필터 교체 (KF $\to$ SVSF)
기존 칼만 필터(KF)를 **SVSF**로 교체하여 모델링 오차에 대한 강건성을 확보합니다.
- **KF:** 가우시안 오차 가정, 부드럽지만 급격한 변화에 느림.
- **SVSF:** 오차의 부호(Sign)와 경계층(Boundary Layer)을 사용하여, 예측이 빗나갈 때 강력하게 보정함 (Sliding Mode Control 원리).

### ③ IMM 구성 (Mixing Strategy)
3개의 SVSF 필터를 병렬로 구성하여 주행 모드를 확률적으로 추정합니다.
- **Model 1 (Straight):** 가속도($\ddot{x}$) $\approx 0$, SVSF 게인 작게 설정 (직진 안정성).
- **Model 2 (Left Switch):** 가속도($\ddot{x}$) $< 0$, SVSF 게인 크게 설정 (빠른 좌분기 반응).
- **Model 3 (Right Switch):** 가속도($\ddot{x}$) $> 0$, SVSF 게인 크게 설정 (빠른 우분기 반응).

---

## 3. AI 코딩 어시스턴트용 명령 프롬프트 (CLI Prompt)

아래 내용을 복사하여 Cursor, Copilot 등의 AI 도구에 입력하세요.

> **[Role]**
> 너는 자율주행 및 제어 이론 전문가야. 현재 `videoAssessor_phase5_IMM.py` 파일은 단일 다항식 피팅(Phase 4) 코드이다. 이를 **"IMM-SVSF (Interacting Multiple Model Smooth Variable Structure Filter)"** 기반의 트래커로 업그레이드해줘.
>
> **[Task Requirements]**
> 1. **`SVSF` 클래스 구현:**
>    - 표준 칼만 필터 대신, 업로드된 논문의 **Smooth Variable Structure Filter** 로직을 따르는 필터 클래스를 만들어라.
>    - **상태 변수:** $[x, \dot{x}, \ddot{x}]$ (위치, 기울기, 곡률)
>    - **Update 단계:** 논문의 수식을 참고하여 **SVSF Gain**($K_{SVSF}$)을 계산하고 상태를 갱신해라. (SVSF의 핵심인 `saturation` 함수와 `boundary layer` 파라미터 $\psi$를 포함할 것)
>
> 2. **`IMMRailTracker` 클래스 구현 (기존 `PolynomialRailTracker` 대체):**
>    - **3-Model 구성:**
>      - Model 1 (Straight): $\ddot{x} \approx 0$ 인 SVSF
>      - Model 2 (Left Switch): $\ddot{x} < 0$ (좌회전 가속도) 인 SVSF
>      - Model 3 (Right Switch): $\ddot{x} > 0$ (우회전 가속도) 인 SVSF
>    - **IMM 로직:**
>      - **Mixing:** 이전 스텝의 확률을 기반으로 상태와 공분산 혼합.
>      - **Mode Probability:** 측정값(Residue)의 우도(Likelihood)를 계산하여 각 모델의 확률 갱신.
>      - **Output:** 확률 가중 평균을 통한 최종 $x$ 위치 도출.
>
> 3. **통합 및 실행:**
>    - `process_frame` 함수에서 Bottom-Up 루프를 돌 때, 단순 피팅 대신 `IMMRailTracker.update(measurement)`를 호출하여 Ego-Track을 추적하도록 변경해라.
>    - 분기점 후보가 여러 개일 때는 `Gating`을 통해 예측값과 가장 가까운 점을 입력으로 사용해라.

---

## 4. 핵심 수식 가이드 (Mathematical Reference)

구현 시 참고해야 할 SVSF의 핵심 수식입니다.

### 1. A priori State Estimate & Error
$$\hat{x}_{k|k-1} = A \hat{x}_{k-1|k-1} + B u_{k-1}$$
$$e_{k|k-1} = z_k - H \hat{x}_{k|k-1}$$

### 2. SVSF Gain Calculation
SVSF 게인은 오차의 크기와 평활화 경계층(Boundary Layer)에 의해 결정됩니다.

$$K_{k} = \text{diag}(|e_{k|k-1}| + \gamma |e_{k-1|k-1}|) \circ \text{sat}(\bar{\psi}^{-1} e_{k|k-1})$$

- $\gamma$: 수렴 속도 계수 ($0 \le \gamma < 1$)
- $\psi$: 평활화 경계층 너비 (Chattering 방지)
- $\text{sat}(\cdot)$: 포화 함수 (Saturation Function)

### 3. State Update
$$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k$$
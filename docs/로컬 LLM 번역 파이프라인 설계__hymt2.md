# 로컬 LLM 번역 파이프라인 설계 — Hy-MT2 (vLLM)

`docs/idea/번역 파이프라인 조사__gpt 대화로그.md`의 일반론을 본 파이프라인 실정에 맞게
재구성한 실행 설계. 2026-10-30 기준.

## 1. 일반론 → 우리 상황 재구성

| 일반론(GPT 문서) | 우리 상황 | 반영 |
|---|---|---|
| 캐릭터 스타일 시트, 화자 정보 | **없음** (OCR 세그먼트만 있음) | 제외. 대신 공식 프롬프트의 배경텍스트 모드만 향후 후보 |
| 기존 번역 로그 활용 | **585K 쌍 코퍼스** (`.omc/research/hdtrans-poc/corpus_pairs.tsv` — 원문×GLM 다수역×빈도×일관성) | TM(정확일치 재사용) + 용어집 주입 + 평가셋 3용도로 전량 활용 |
| 유사일치(fuzzy) 참고 제공 | 오탐 위험 + 구현 비용 | v1 제외. 정확일치만 |
| 1차 소형 → 규칙 검증 → 상위 재번역 | 동일 채택. 상위 = **동일 계열** Hy-MT2-7B-FP8 (이종 Qwen 등 조합 비용↑) | 3티어: TM → 1.8B → 7B (+GLM 최후 비상용) |
| 온도 0.1~0.3 (일반 LLM 기준) | Hy-MT2 **공식 권장값 별도** (모델 전용 설정 우선) | 1.8B·7B 동일: t0.7/p0.6/k20/rep1.05 |
| 외부 API 비용/429 관리 | 로컬 vLLM — 레이트리밋 없음 | 적응 동시성 불필요. 고정 세마포어 동시성 |

핵심 목표: **"가능한 한 1.8B 선에서 끝내기"** — 측정 지표 `% resolved by ≤1.8B` 를
페이지 단위 로그로 상시 추적. 7B 트래픽은 곧 품질 게이트 통과율의 역수.

## 2. 리소스 예산 (RTX 4090 24GB, RAM 94GB, 여유 VRAM ~20.7GB)

| 서비스 | 모델 | VRAM | RAM | 기동 |
|---|---|---|---|---|
| hymt2-small (기본 단독) | Hy-MT2-1.8B-FP8 | ~3–4GB (KV캐시 포함) | ~1GB | 파이프라인과 동시 상시 |
| hymt2-big (수리 패스) | Hy-MT2-7B-FP8 (dense) | ~10.8GB (util 0.45) | ~1GB | **배타 모드** — `./hymt2_repair.sh` 스왑 시에만 (§3.6) |
| GLM 폴백 | (외부 API) | 0 | 0 | big 다운 + env 설정 시만 |

- 4090 = Ada → FP8 가속 지원. 1.8B-FP8 가중치 ~1.9GB.
- **30B-A3B-FP8는 이 환경에서 기동 불가로 폐기(2026-10-30)** — MoE 워크스페이스 할당에서
  CUDA illegal access(TRITON·Marlin 백엔드 모두, `--cpu-offload-gb`·util 0.55/0.7 무관).
  WSL2가 pinned memory를 지원하지 않아 UVA 오프로더가 디바이스 메모리로 폴백하는 것이
  근접 원인. dense인 7B-FP8(≈7.5GB)은 오프로드 없이 상주하며 이 문제와 무관.
- 스케일업 경로: big 서비스를 올리거나 gpu_pre 컨테이너 수를 조정하는 것만으로
  증감 — 번역기 코드 변경 불필요 (URL/env 로 분리).

## 3. 아키텍처

```
세그먼트(페이지 단위 유입, finish.py Pass B)
 │
 ├─ Tier 0   TM 정확일치 — norm 키 조회 (configs/hymt2-tm.tsv, 코퍼스 share≥0.75·빈도≥3)
 │           1차: 전체 norm 키 → 2차: 행끝 부호(?!.…~♥) 제거 키
 │           히트: LLM 호출 0회, GLM 다수역 품질 그대로 재사용
 │           측정(2026-10-30): 4,464 키 — 발생량 기준 10.0% / 유니크 원문 0.8% 커버
 ├─ Tier 0.5 벡터 근접검색 (§3.5) — 임베딩: df.flaglow.cc /embedder (bge-m3-mlx-8bit)
 │           sim≥0.985 且 DB행 share≥0.75 → GLM 다수역 직접 재사용 (LLM 0회)
 │           0.88≤sim<0.985 → 공식 용어집 형식으로 근접 쌍을 1.8B 프롬프트에 참조 주입
 ├─ Tier 1   Hy-MT2-1.8B-FP8 — 미스 세그먼트 전수를 동시성 16으로 1차 번역
 │           (공식 프롬프트, 세그먼트별 독립 요청 — vLLM 연속 배칭이 처리량 담당)
 ├─ 게이트   로컬 규칙 검증: 빈 출력 / 한글 부재 / 원문 2배 과길이 / 원문 통과(미번역)
 │           실패 시 1.8B에 힌트 부여 1회 재시도
 └─ Tier 2   Hy-MT2-7B-FP8 — 잔여 실패 세그먼트만 2차 (소형 전수 선처리 후 일괄)
             big 불가 시 → GLM 1회(설정된 경우) → 빈 칸
```

- **스케줄링(소형 선처리)**: `_translate` 1회 호출 안에서 [1.8B 전수 병렬] → [게이트] →
  [7B 실패분 일괄] 순서. 소형 작업이 대형 작업에 선점되지 않는 구조.
- **세그먼트별 독립 요청**(GLM 경로의 `<|N|>` 배치와 다름): 1.8B급 마커 규율 리스크
  제거, 실패 격리, 에스컬레이션 단위 자연스러움. vLLM은 단발 다수 동시 요청에 최적.
- **용어집 주입**: 정적 사전(configs/hdtrans-userdict.txt 형식 TSV)에서 현재 세그먼트에
  부분문자열로 걸리는 키만 추려(상한 8개) 공식 용어집 프리픽스
  `Reference the following translations:` 로 주입.

## 3.5 벡터 근접검색 (Tier 0.5)

GPT 문서의 "유사일치는 자동 적용 금지, 참고로만" 원칙을 임베딩 유사도로 구현한 티어.

- **임베딩**: `df.flaglow.cc /embedder/v1` (OpenAI 호환, 자가 서비스) — `bge-m3-mlx-8bit`
  (1024차원, 다국어 CJK). 빌드와 쿼리가 같은 모델이어야 하므로 `BUILD.json`의 모델명과
  `HYMT2_EMB_MODEL`이 다르면 티어가 자동 비활성된다.
- **인덱스**: numpy 평탄 행렬(`configs/hymt2-vec/emb.npy`, fp16 저장·fp32 상주 정규화).
  45K×1024 matvec < 1ms — faiss/pgvector 같은 외부 인프라 불필요(신규 의존 0,
  RAM ~186MB/프로세스). 백만 규모로 늘어나는 시점에만 faiss 교체를 재검토.
- **대상 코퍼스**: count≥2·share≥0.5 → 유니크 norm 키 ~45.5K (발생량의 20.5%).
  share 0.5 미만 행은 GLM 스스로 불일관하는 원문 — 참조 재료로도 부적합해 제외.
- **2단 임계값** (env `HYMT2_VEC_REUSE_SIM`/`HYMT2_VEC_REF_SIM`):
  - 재사용 ≥0.985 + DB행 share≥0.75 — TM과 동일 품질 기준으로 직접 채택.
    BGE-M3는 짧은 일본어 감탄사군이 0.91~0.98에 밀집하므로(2026-10-30 표본 보정,
    최빈 200키 NN-sim p50 0.911/p95 0.979) 0.98 미만은 직접 채택하지 않는다.
  - 참조 0.88~0.985 — 근접 원문→GLM 다수역 쌍을 Hy-MT2 공식 용어집 블록에 주입
    (용어집과 병합, 상한 8쌍). 문맥 없는 1.8B에게 작품 규율을 주는 핵심 경로.
- **운영**: 페이지 단위로 미스 세그먼트 전체를 1회 배치 임베딩(64개/요청, 병렬) —
  API 죽으면 해당 티어만 건너뛰고 1.8B로 직행(경고 1행). 빌더·보정 절차는 §8.
- **후보(v2)**: `df.flaglow.cc /reranker` (bge-reranker-v2-m3)로 참조 top-K 재순위 —
  임계값 하위 후보들까지 안전하게 소비 가능.

## 3.6 배타 모드 — 로컬 LLM 한 번에 한 모델 (기본)

두 vLLM 서버(1.8B 상시 + 7B 필요시)를 동시에 올리면 VRAM·전력 부담이 크다.
기본 정책(`HYMT2_EXCLUSIVE=1`)은 **즉시 상향을 금지**하고 아래 흐름으로 대체한다:

```
번역 중 소형 게이트 실패 → 스플 적립 (tmp/hymt2-spill/pending.jsonl, 원문+참조 보존)
                             → 빈 칸/폴백으로 렌더링은 계속
./hymt2_repair.sh           → 소형 정지 → 7B 기동(profile escalation) → 헬스 대기 →
                              스플 일괄 수리(manga_translator/translators/hymt2_repair) →
                              성공은 캐시(tmp/hymt2-cache/big.jsonl)에, 실패는 스플 유지 →
                              7B 정지 → 소형 재기동 (실패 시에도 소형 복구 보장)
이후 번역                    → Tier 0.6 수리 캐시 조회가 빈 칸을 채움 (재렌더링 대상만)
```

- 스플/캐시는 `tmp/` 에 마운트 기반 공유 — finish 컨테이너·수리 컨테이너·호스트가
  같은 파일을 본다. 옵션: `HYMT2_EXCLUSIVE=0`이면 기존 즉시 상향(7B 동시 기동 운용).
- 빈 칸으로 렌더링된 페이지는 수리 후 `--overwrite` 재렌더링으로 완성한다
  (finish.py는 dest 존재 시 스킵이 기본).

## 4. 모델 전용 설정 (Hy-MT2 공식)

- 프롬프트(EN 기본형, 대상 언어는 **전체 언어명** 사용):
  `Translate the following text into Korean. Note that you need only output the
  translated result without any additional explanation:\n\n{source}`
- 시스템 프롬프트 **없음**(공식 권장).
- 샘플링: 1.8B `temperature=0.7, top_p=0.6, top_k=20, repetition_penalty=1.05`
  / 7B는 1.8B와 동일 샘플링, max_context 8192.
- OpenAI 호환 `/v1/chat/completions` — vLLM 서빙과 그대로 대응.

## 5. 배포 (docker compose — `demo/doc/docker-compose-hymt2.yml`)

- `hymt2-small`: `vllm/vllm-openai` 이미지, 포트 **8011**, GPU 전체 허용.
- `hymt2-big`: 동일 이미지, 포트 **8012**, `--cpu-offload-gb 18`,
  profile `escalation` (기본 미기동: `docker compose --profile escalation up`).
- 모델 캐시 볼륨 공유(`~/.cache/huggingface` 마운트) — 재다운로드 방지.
- 라이브 라인(manga_image_translator_gpu1~5)과는 별도 컴포즈 — GPU 컨테이너 재시작 없음.

## 6. 설정·선택 방법

- **토글**: `configs/hymt2-kor.toml` 선택(또는 GUI 설정 드롭다운) — 기존
  gpt-kor.toml(GLM)/hd-kor.toml 경로는 그대로. `[translator] translator = "hymt2"`.
- **재정의(env)**: `HYMT2_SMALL_BASE`(기본 `http://localhost:8011/v1`),
  `HYMT2_BIG_BASE`(기본 `http://localhost:8012/v1`, 빈 값=에스컬레이션 없음),
  `HYMT2_SMALL_MODEL`/`HYMT2_BIG_MODEL`(served-model-name),
  `HYMT2_CONCURRENCY`(16), `HYMT2_TIMEOUT`(30s), `HYMT2_TM`(경로, 빈 값=TM 끔),
  `HYMT2_GLOSSARY`(경로, 기본 hdtrans-userdict.txt, 빈 값=끔),
  `HYMT2_GL_FALLBACK`(1=CUSTOM_OPENAI_* 환경으로 GLM 비상 폴백),
  `HYMT2_EMB_BASE`(기본 `https://df.flaglow.cc/embedder/v1`)/`HYMT2_EMB_MODEL`
  (bge-m3-mlx-8bit)/`HYMT2_VEC_DIR`(기본 configs/hymt2-vec, 없거나 빈 값=티어 끔),
  `HYMT2_VEC_REUSE_SIM`(0.97)/`HYMT2_VEC_REF_SIM`(0.88)/`HYMT2_VEC_K`(3,
  참조 상위 k개)/`HYMT2_REF_F1_GATE`(0.3, 출력-참조 바이그램 F1 하한 — 미달 시 7B
  상향, 0=끔)/`HYMT2_DIRECT_BIG`(0, 무참조 세그먼트 7B 직행),
  `HYMT2_EXCLUSIVE`(1=로컬 LLM 한 번에 한 모델·스필/수리 흐름, 0=즉시 상향),
  `HYMT2_CACHE`/`HYMT2_SPILL`(기본 tmp/hymt2-cache/big.jsonl · tmp/hymt2-spill/pending.jsonl).
- 번역기 모듈은 `translators/` 디렉터리 마운트로 finish 컨테이너에 자동 반영
  (translate_gui.py FINISH_MOUNTS).

## 7. 품질 게이트 상세 (custom_openai 패턴 계승)

1. 빈 출력 → 재시도
2. 한글 미포함(자음+모음 결합 완성형 기준) → 재시도 — 소형 모델의 원문 통과/영어 회귀 감지
3. 원문 2배 이상 과길이(`_LENGTH_HARD_CAP=2` 계승, 원문 1자 예외) → 힌트 재시도 → 최단 후보 차선
4. 출력==원문(미번역 통과) → 에스컬레이션
5. 최종 스윕: 게이트 통과 못 한 세그먼트는 7B → (옵션)GLM → 빈 칸 (거부 문구 렌더링 방지)

## 8. 평가·운영

### 1차 실측 (2026-10-30, vLLM 0.32.0 eager · 계층샘플 700 — A:TM키 150/B:벡터DB 비TM 350/C:GLM불일관층 200)

| 지표 | 1.8B 단독 | 7B 단독 | 1.8B+게이트(v3) | GLM(현행) |
|---|---|---|---|---|
| 페이지(25세그) 지연 | **2.4~2.8s** | 3.4s | ~2.6s+수리 | 12.2s |
| 처리량 | 9~10 seg/s | 7.4 seg/s | — | ~2 seg/s |
| API 토큰/세그 | 0 | 0 | 0 | ~85 |
| TM키층(A) 일치 | 97.3% | 97.3% | 97.3% | — |
| B층 정확일치 / F1 / F1≥0.6 | 8.0% / 39.6 / 24.5% | **10.6% / 47.0 / 35.0%** | 7.4% / 39.5 / 23.8% | 자기일관성 31% |
| C층 정확일치 / F1 / F1≥0.6 | 4.5% / 34.7 / 25.6% | 5.8% / 36.5 / 27.9% | 5.5% / 35.2 / 26.1% | — |

- **계층별 근거(f1_by_tier)**: 참조주입이 양 모델에서 +6~12pp — 참조가 품질의 핵심 레버.
  7B의 1.8B 대비 우위는 B층 무참조(+8.6pp)에 집중, C층 무참조는 역전 →
  무조건 직행은 기본 금지(HYMT2_DIRECT_BIG 옵션).
- **v3(1.8B 우선 + 재사용 0.97 + 참조 k=3 + 앵커이탈 F1 게이트 0.3)**:
  벡터 재사용 2→9건, 게이트가 25건(약 8%)을 7B로 상향. 상향 25건 짝 비교에서
  개선 9 / 동일 12 / 악화 4 — `知るかよ→내가 알게 뭐야.`(F1 0→0.91) 등 표적 효과.
  다만 t=0.7 샘플링 변동으로 B층 F1 총량은 1.8B 단독과 동일 수준(±노이즈).
- **운영 딜**: 품질 상한은 7B 전수(47.0), 원가·속로는 1.8B 우선+게이트. 게이트 임계와
  직행 옵션으로 중간 지점 조정 가능.

- 7B는 1.8B 대비 B층 F1 +7.4pp·근접비중 +10.5pp — **표현 관례가 GLM에 실제로 더 가깝다**.
  다만 정확일치 기준 격차는 여전해, "GLM 다수역과 동일 문장"은 GLM 자기일관성(31%)이 천장.

- 1.8B는 유창·한글율 99%+·길이비 0.95~1.05로 게이트를 잘 통과하지만, **표현이 GLM
  관례와 달라지는 비중이 크다** (GLM 자기일관성 31%가 동일 레퍼런스·동일 지표의
  실용 천장 — 다양한 유효 번역 존재 시 정확일치는 과소평가).
- 스타일 지시(공식 【...】 형식, 반말 대사체)로 v1의 해요체 계통 편차는 교정됨
  (정성: `どうかな？` 부디 어떠신가요?→어떻게 생각해?). 정확일치율 변화 미미.
- 티어 실측: TM이 B의 4.9%/C의 11.5%까지 흡수(부호변형 2단 키), 벡터 재사용은
  0.985 기준 보수적(B 2/C 3건), 참조주입 B 34%/C 55%.
- **소형 단독으로 GLM 대체는 미달** — 7B 에스컬레이션으로 B층 F1 39.6→47.0 상승을
  확인. 다음 관문: ① few-shot k=2~3·vec 임계 완화, ② 연질(chrF) 기반 게이트
  (F1 낮은 세그먼트만 7B 직행), ③ 최종은 인간 리뷰 샘플링.
- vLLM 운영 지식: FP8+Ada 조합 CUDA 그래프 캡처 오류 → `--enforce-eager`로 해결
  (1.8B 성능 영향 미미). WSL 방화벽으로 브리지 게이트웨이→호스트 차단 →
  명명 네트워크 `hymt2`로 finish 컨테이너 직접 attach (translate_gui FINISH_NETWORK).

### 파일럿 (2026-10-30, hd-pilot 14페이지 239세그먼트 — 배타 흐름 전체 실증)

- 1차 렌더 18.6s(1.8B 단독) → 스플 13 → 수리 스왑(12 성공) → 재렌더(수리캐시 히트) →
  잔여 3건 2차 수리(4건) → 최종 재렌더: **빈칸 0·스필 0 수렴**.
- 티어 분포: 1.8B 78% (무참조 61% + 참조주입 18%), TM 14%, 7B 수리 7%, 벡터재사용 0.4%
  — ≤1.8B+TM 해결률 약 93%.
- 정성: `今って大丈夫ですか？→지금은 괜찮아요?` 등 GLM과 근접 다수. 일부 어색
  (`あっいらっしゃい→어-이리 오세요` — 인사말 관례) — 인간 리뷰 판정 대상.
- 파일럿이 잡은 결함: 페이지 병렬 `_stat` 레이스(호출별 로컬 통계로 수정),
  중첩 bind-mount가 root 소유 tmp/ 를 만드는 문제(호스트 사전 생성으로 회피 —
  운영 finish 레인은 tmp/ 통째 마운트라 무관).
- 운용 권장: 파일럿 절차 = 렌더 → `./hymt2_repair.sh` → 재렌더(--overwrite) 반복,
  스플이 비면 완성. GLM 병행 대비 검수 포인트는 어색 관용구·호칭·의성어.


- 회귀 평가셋: 코퍼스에서 빈도 계층화 샘플링 → 1.8B/7B 번역 vs GLM 다수역
  (정확일치율, 한글율, 길이비) — `.omc/research/hymt2-poc/` 에서 hdtrans-poc
  절차 계승. GLM 자기일관성(57.3%)이 상한 참고선.
- 페이지 로그 1행 요약: `tm / 벡터재사용 / 1.8B(참조주입·재시도·실패) / 7B / GLM /
  빈칸 | ≤1.8B 해결률` — "% resolved by ≤1.8B" 운영 지표.
- TM 갱신: 코퍼스 재마이닝(`.omc/research/hdtrans-poc/extract_segments.py` +
  `build_tm.py`) 후 tsv 재생성 — 파일 헤더에 절차 기록.
- 벡터 DB 갱신: 같은 코퍼스로 `build_vec.py --full` (임베딩 ~45K 키, 체크포인트
  재개 지원) → `configs/hymt2-vec/{emb.npy, meta.tsv, BUILD.json}` 재생성.
  git 제외(.gitignore) — configs/ 마운트로 컨테이너에 자동 반영.

## 9. v2 후보 (v1 범위 밖)

- 페이지 맥락 주입(공식 background_text 모드) — 직전 페이지 번역 누적 활용
- fuzzy(95–99%) 참고 번역 제공
- 7B 콜드스타트 자동화(GUI 오토스케일러 패턴 이식, 대기열 임계시 기동)
- 세그먼트 난이도 사전분류(코퍼스 GLM 일관성 낮은 원문 = 애초에 7B 직행)

# 로컬 LLM 번역 파이프라인 설계 — Hy-MT2 (vLLM)

`docs/idea/번역 파이프라인 조사__gpt 대화로그.md`의 일반론을 본 파이프라인 실정에 맞게
재구성한 실행 설계. 2026-10-30 기준.

## 1. 일반론 → 우리 상황 재구성

| 일반론(GPT 문서) | 우리 상황 | 반영 |
|---|---|---|
| 캐릭터 스타일 시트, 화자 정보 | **없음** (OCR 세그먼트만 있음) | 제외. 대신 공식 프롬프트의 배경텍스트 모드만 향후 후보 |
| 기존 번역 로그 활용 | **585K 쌍 코퍼스** (`.omc/research/hdtrans-poc/corpus_pairs.tsv` — 원문×GLM 다수역×빈도×일관성) | TM(정확일치 재사용) + 용어집 주입 + 평가셋 3용도로 전량 활용 |
| 유사일치(fuzzy) 참고 제공 | 오탐 위험 + 구현 비용 | v1 제외. 정확일치만 |
| 1차 소형 → 규칙 검증 → 상위 재번역 | 동일 채택. 상위 = **동일 계열** Hy-MT2-30B-A3B-FP8 (이종 Qwen 등 조합 비용↑) | 3티어: TM → 1.8B → 30B (+GLM 최후 비상용) |
| 온도 0.1~0.3 (일반 LLM 기준) | Hy-MT2 **공식 권장값 별도** (모델 전용 설정 우선) | 1.8B: t0.7/p0.6/k20/rep1.05 · 30B: t0.7/p1.0/rep1.0 |
| 외부 API 비용/429 관리 | 로컬 vLLM — 레이트리밋 없음 | 적응 동시성 불필요. 고정 세마포어 동시성 |

핵심 목표: **"가능한 한 1.8B 선에서 끝내기"** — 측정 지표 `% resolved by ≤1.8B` 를
페이지 단위 로그로 상시 추적. 30B 트래픽은 곧 품질 게이트 통과율의 역수.

## 2. 리소스 예산 (RTX 4090 24GB, RAM 94GB, 여유 VRAM ~20.7GB)

| 서비스 | 모델 | VRAM | RAM | 기동 |
|---|---|---|---|---|
| hymt2-small (상시) | Hy-MT2-1.8B-FP8 | ~3–4GB (KV캐시 포함) | ~1GB | 파이프라인과 동시 상시 |
| hymt2-big (에스컬레이션) | Hy-MT2-30B-A3B-FP8 (MoE, 활성 3B) | ~12–14GB (`--cpu-offload-gb` 로 일부 전문가 CPU 이관) | +~18GB | compose profile `escalation` — 필요시만 |
| GLM 폴백 | (외부 API) | 0 | 0 | big 다운 + env 설정 시만 |

- 4090 = Ada → FP8 가속 지원. 1.8B-FP8 가중치 ~1.9GB.
- 30B-A3B-FP8 가중치 ~30GB > VRAM → vLLM `--cpu-offload-gb` 로 MoE 전문가 일부를
  CPU로 보낸다. 활성 파라미터가 톤당 3B뿐이라 throughput 손실이 제한적.
- 스케일업 경로: big 서비스를 올리거나 gpu_pre 컨테이너 수를 조정하는 것만으로
  증감 — 번역기 코드 변경 불필요 (URL/env 로 분리).

## 3. 아키텍처

```
세그먼트(페이지 단위 유입, finish.py Pass B)
 │
 ├─ Tier 0  TM 정확일치 — norm 키 조회 (configs/hymt2-tm.tsv, 코퍼스 share≥0.75·빈도≥3)
 │           1차: 전체 norm 키 → 2차: 행끝 부호(?!.…~♥) 제거 키
 │           히트: LLM 호출 0회, GLM 다수역 품질 그대로 재사용
 │           측정(2026-10-30): 4,464 키 — 발생량 기준 10.0% / 유니크 원문 0.8% 커버
 ├─ Tier 1  Hy-MT2-1.8B-FP8 — 미스 세그먼트 전수를 동시성 16으로 1차 번역
 │           (공식 프롬프트, 세그먼트별 독립 요청 — vLLM 연속 배칭이 처리량 담당)
 ├─ 게이트  로컬 규칙 검증: 빈 출력 / 한글 부재 / 원문 2배 과길이 / 원문 통과(미번역)
 │           실패 시 1.8B에 힌트 부여 1회 재시도
 └─ Tier 2  Hy-MT2-30B-A3B-FP8 — 잔여 실패 세그먼트만 2차 (소형 전수 선처리 후 일괄)
             big 불가 시 → GLM 1회(설정된 경우) → 빈 칸
```

- **스케줄링(소형 선처리)**: `_translate` 1회 호출 안에서 [1.8B 전수 병렬] → [게이트] →
  [30B 실패분 일괄] 순서. 소형 작업이 대형 작업에 선점되지 않는 구조.
- **세그먼트별 독립 요청**(GLM 경로의 `<|N|>` 배치와 다름): 1.8B급 마커 규율 리스크
  제거, 실패 격리, 에스컬레이션 단위 자연스러움. vLLM은 단발 다수 동시 요청에 최적.
- **용어집 주입**: 정적 사전(configs/hdtrans-userdict.txt 형식 TSV)에서 현재 세그먼트에
  부분문자열로 걸리는 키만 추려(상한 8개) 공식 용어집 프리픽스
  `Reference the following translations:` 로 주입.

## 4. 모델 전용 설정 (Hy-MT2 공식)

- 프롬프트(EN 기본형, 대상 언어는 **전체 언어명** 사용):
  `Translate the following text into Korean. Note that you need only output the
  translated result without any additional explanation:\n\n{source}`
- 시스템 프롬프트 **없음**(공식 권장).
- 샘플링: 1.8B `temperature=0.7, top_p=0.6, top_k=20, repetition_penalty=1.05`
  / 30B `temperature=0.7, top_p=1.0, repetition_penalty=1.0`, max_context 8192.
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
  `HYMT2_GL_FALLBACK`(1=CUSTOM_OPENAI_* 환경으로 GLM 비상 폴백).
- 번역기 모듈은 `translators/` 디렉터리 마운트로 finish 컨테이너에 자동 반영
  (translate_gui.py FINISH_MOUNTS).

## 7. 품질 게이트 상세 (custom_openai 패턴 계승)

1. 빈 출력 → 재시도
2. 한글 미포함(자음+모음 결합 완성형 기준) → 재시도 — 소형 모델의 원문 통과/영어 회귀 감지
3. 원문 2배 이상 과길이(`_LENGTH_HARD_CAP=2` 계승, 원문 1자 예외) → 힌트 재시도 → 최단 후보 차선
4. 출력==원문(미번역 통과) → 에스컬레이션
5. 최종 스윕: 게이트 통과 못 한 세그먼트는 30B → (옵션)GLM → 빈 칸 (거부 문구 렌더링 방지)

## 8. 평가·운영

- 회귀 평가셋: 코퍼스에서 빈도 계층화 샘플링 → 1.8B/30B 번역 vs GLM 다수역
  (정확일치율, 한글율, 길이비) — `.omc/research/hymt2-poc/` 에서 hdtrans-poc
  절차 계승. GLM 자기일관성(57.3%)이 상한 참고선.
- 페이지 로그 1행 요약: `tm=N small=OK/FAIL big=OK glm=OK` — "% resolved by ≤1.8B" 운영 지표.
- TM 갱신: 코퍼스 재마이닝(`.omc/research/hdtrans-poc/extract_segments.py` +
  `build_tm.py`) 후 tsv 재생성 — 파일 헤더에 절차 기록.

## 9. v2 후보 (v1 범위 밖)

- 페이지 맥락 주입(공식 background_text 모드) — 직전 페이지 번역 누적 활용
- fuzzy(95–99%) 참고 번역 제공
- 30B 콜드스타트 자동화(GUI 오토스케일러 패턴 이식, 대기열 임계시 기동)
- 세그먼트 난이도 사전분류(코퍼스 GLM 일관성 낮은 원문 = 애초에 30B 직행)

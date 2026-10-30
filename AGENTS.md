# AGENTS.md — 에이전트 작업 규칙

이 저장소는 zyddnys/manga-image-translator의 fork를 **상시 운영 중인 일본어→한국어 만화 대량 번역 파이프라인**으로 개조한 워크스페이스다. 일반적인 오픈소스 기여 매너보다 아래 운영 규칙이 우선한다.

## 절대 규칙 (라이브 라인 보호)

- `python3 translate_gui.py` (8787 대시보드), docker 컨테이너 `manga_image_translator_gpu1~5`, `mit-finish-*`는 **상시 실행 중**이다. `pkill python`·`docker stop`·컨테이너 재시작 등 임의 종료는 진행 중 번역을 파괴한다 — 종료는 반드시 `./shutdown.sh` 절차로만.
- 대시보드(`translate_gui.py`) 재시작은 고아게이트(`worker_loop`의 `pgrep` 검사)가 컨테이너 내 고아 프로세스를 보호하므로 비교적 안전하지만 실행 중 작업의 추적이 끊긴다. 사용자에게 확인받고 실행할 것.
- 번역 결과·캐시는 디스크에 즉시 저장되고 `./startup.sh --resume <날짜폴더>`로 이어할 수 있다 — 뭔가 잘못됐으면 재시작보다 resume 흐름을 활용한다.

## 커밋 제외 경로

`data/`, `translated/`, `tmp/`, `result/`, `.env`, `.omc/`, `.omx/`, `.gpu_count`는 커밋하지 않는다 (`.gitignore`에 반영됨). `.env`에는 LLM API 키(`OPENAI_COMPATIBLE_*`)가 있다.

## 환경 주의사항

- **로컬 `.venv`는 깨져 있다** (디렉터리 마이그레이션로 절대 경로 심링크 무효). 파이썬 테스트는 실제 파이프라인이 쓰는 이미지에서 실행한다:
  ```bash
  docker run --rm --entrypoint python -v "$PWD:/app" -w /app zyddnys/manga-image-translator:main <스크립트>
  ```
  (이미지 기본 엔트리포인트가 CLI라 `--entrypoint python` 재정의 필수)
- 순수 표준라이브러리 파일(`translate_gui.py` 등)은 `python3 -m py_compile` + 임시 인스턴스(다른 포트) 기동으로 검증한다.
- 코드 수정은 컨테이너에 바인드마운트되어 있어 **다음 작업부터 자동 반영**된다 (`custom_openai.py`, `keys.py`, `args.py`, `manga_translator.py`, `mode/local.py`, `mode/finish.py`). 단 `translate_gui.py`는 재시작해야 반영된다.
- GPU 컨테이너의 상주 서버(`manga_translator shared`)는 인프라 프로세스다 — busy 판정은 `manga_translator local`/`mode.finish` 패턴으로만 한다 (광범위 패턴 쓰면 모든 컨테이너가 영구 busy가 된다).

## 운영 스택 구조

- `translate_gui.py` — 대시보드 + HTTP API(8787). **systemd 유저 서비스로 상주**(`systemctl --user start|stop translate-gui`, Restart=always). 작업 큐, 2레인(GPU: pre/full · 독립 CPU 컨테이너: finish), GPU 오토스케일러(목표 개수 1~5, 기본 3, `.gpu_count`에 저장, 20초 주기). API 명세: `GUI_API.md`
- `manga_translator/mode/finish.py` — Pass B. GPU 캐시(`tmp/gpu_pre/<rel>/{png,pkl}`) 기반 번역·렌더링 (GPU 불필요)
- `manga_translator/translators/custom_openai.py` — GLM 번역기. 세그먼트별 품질 게이트(번역 거부 감지·재시도, 원문 2배 길이 하드캡, 정상 세그먼트 보존), 429 적응 동시성
- `manga_translator/translators/hymt2.py` — Hy-MT2 로컬 LLM 번역기(vLLM). TM 정확일치 → 벡터 근접검색(df.flaglow.cc 임베딩 + numpy 평탄 인덱스, `configs/hymt2-vec/` — git 제외) → 수리 캐시 → 1.8B-FP8 전수 → 게이트 → 7B-FP8 에스컬레이션(+GLM 비상 폴백). `configs/hymt2-kor.toml`로 선택, `HYMT2_*` env로 재정의. 설계: `docs/guides/hymt2-pipeline-design.md`
- `./hymt2_repair.sh` — 배타 모드 수리 패스(기본 `HYMT2_EXCLUSIVE=1`): 로컬 LLM 동시 1개 원칙. 소형 정지 → 7B 기동 → 스플(`tmp/hymt2-spill/`) 일괄 수리 → 캐시(`tmp/hymt2-cache/`) 저장 → 소형 재기동
- `configs/` — `gpt-kor.toml`(번역), `pass_a.toml`(pre 전용), `book-ocr.toml`(OCR), `hd-kor.toml`(hdTrans), `hymt2-kor.toml`(Hy-MT2 로컬 LLM)
- `demo/doc/docker-compose-web-with-gpu.yml` — GPU 컨테이너 5서비스 (gpu1~5)
- 기동/종료: `./startup.sh [--resume <폴더>...]` / `./shutdown.sh` — startup이 vLLM 소형(hymt2-small, 전처리 GPU와 분리된 고정 LLM 컨테이너)도 함께 기동. **전처리 GPU 컨테이너는 1대(gpu1)가 기본 원칙** (`docs/decisions/0001-gpu-single-container-policy.md` — VRAM당 처리량 4.3 p/분/GB로 최적, 3대는 +53%에 그침). 상한만 vLLM 감지 시 3(`GPU_MAX_WITH_LLM`)
- finish 레인 동시성: env `FINISH_CONCURRENCY` — 로컬 LLM(vLLM) 감지 시 기본 10, GLM 운용 시 3(레이트리밋). GLM API 동시성은 `FINISH_API_CONCURRENCY`(기본 3, 레인 수와 분리). GPU 목표 개수 초기값: env `GPU_CONTAINER_COUNT`(기본 3)

## 커밋 컨벤션

Conventional commits(`feat:`, `fix:` …) — upstream 역사 참고. 본문은 한국어로 써도 된다.

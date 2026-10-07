# 만화 번역 대시보드 API

`translate_gui.py` 서버(기본 `http://localhost:8787`)의 HTTP API.
모든 응답은 JSON (`Access-Control-Allow-Origin: *` — 외부 웹앱/스크립트에서 바로 호출 가능).

## 조회

### `GET /api/summary` — 전체 진행 요약 (연동용 첫걸음)

```json
{
  "jobs": {"running": 3, "queued": 1, "done": 12, "error": 0, "canceled": 0},
  "pages": {"done": 626, "total": 1191, "percent": 52.6},
  "active": [
    {"container": "gpu1", "folder": "20261003/[Uten Ameka] ...", "saved": 90, "total": 137}
  ],
  "containers": [{"name": "manga_image_translator_gpu1", "status": "실행중"}]
}
```

### `GET /api/state` — 전체 상태 (작업 목록 포함)

`containers` + `jobs` 배열. 각 작업:

| 필드 | 설명 |
|---|---|
| `id` | 작업 ID (8자리) |
| `folder` | 입력 폴더 (`data/` 기준 상대경로) |
| `dest` | 출력 폴더 (`translated/` 기준, `원본 ｜ 한국어 제목`) |
| `status` | `queued` / `running` / `done` / `error` / `canceled` |
| `saved`, `total` | 완료 페이지 수 / 전체 페이지 수 |
| `current` | 현재 처리 중 페이지 파일명 |
| `duration` | 실행 초 (대기 중이면 null) |
| `error`, `done_note` | 오류 내용 / 완료 요약 (예: "22장 / 3.2분") |

### `GET /api/jobs?id=<ID>&since=<N>` — 작업별 로그

`since`에 마지막으로 받은 `total`을 넘기면 그 이후 줄만 증분 수신.
응답: `{"lines": [...], "total": <누적 줄수>}`

### `GET /api/browse?path=<상대경로>` — 폴더 탐색
### `GET /api/configs` — 설정 파일 목록

## 제어

### 처리 모드 (`mode`)

작업 추가 시 `mode` 필드로 파이프라인 분리 제어 (기본 `full`):

| mode | 실행 레인 | 동작 |
|---|---|---|
| `full` | GPU 컨테이너 | 기존 전체 파이프라인 (검출→OCR→번역→렌더 일괄) |
| `pre` | GPU 컨테이너 | GPU 사전처리만 (검출·OCR·인페인팅) → `tmp/gpu_pre/<rel>/` 캐시. **완료 시 같은 권의 finish가 자동 등록** (2레인 체이닝) |
| `finish` | 독립 CPU 컨테이너 | 캐시 + 원본으로 GLM 번역·렌더링만 (GPU 불필요). 성공 시 캐시 자동 삭제 |

- `pre`는 이미 번역 완료된 권과 캐시 완료 권을 자동 스킵한다 → 대량 재등록 안전
- finish 레인 동시성은 GUI 실행 시 env로 조절: `FINISH_CONCURRENCY`(컨테이너 수, 기본 3), `FINISH_API_CONCURRENCY`(컨테이너당 GLM 동시 요청, 기본 3)

### GPU 컨테이너 목표 개수

GPU 레인(pre/full)에서 유지할 컨테이너 개수. **1~5, 기본 3**. 대시보드 헤더의 `GPU [선택]개` 드롭다운과 동일 기능.

- 작업이 있으면 목표 개수만큼 기동되고, 목표를 넘어 실행 중인 컨테이너는 **작업이 배정되지 않은 것만** 정지된다 (실행 중 작업은 계속 진행되고 종료 후 정리)
- 무작업이 5분(`GPU_IDLE_GRACE`) 지속되면 전체 반납 — 목표 개수는 유지되어 다음 작업 시 그 개수만큼 재기동
- 설정값은 `.gpu_count` 파일에 저장되어 대시보드 재시작에도 유지 (env `GPU_CONTAINER_COUNT`는 초기 기본값)

#### `GET /api/gpu` — 현재 설정 조회

```json
{"count": 3, "max": 5,
 "containers": ["manga_image_translator_gpu1", "...gpu2", "...gpu3", "...gpu4", "...gpu5"],
 "up": ["manga_image_translator_gpu1", "manga_image_translator_gpu2", "manga_image_translator_gpu3"]}
```

#### `POST /api/gpu` — 개수 설정 / 기동 / 반납

```json
{"action": "count", "count": 5}
{"action": "up"}
{"action": "down"}
```

| action | 동작 | 응답 |
|---|---|---|
| `count` | 목표 개수 즉시 적용 (1~5 밖이면 400) | `{"ok": true, "count": 5, "up": [...]}` |
| `up` | 목표 개수만큼 즉시 기동 (idle 타이머 리셋) | `{"ok": true}` |
| `down` | 전체 반납 (실행 중 GPU 작업이 있으면 409) | `{"ok": true}` |

`GET /api/summary`·`/api/state`의 `gpu` 객체에도 `desired`(목표 개수)가 포함된다.

### `POST /api/jobs` — 작업 추가

```json
{"folder": "20261003_ver1/[Author] Title (ID)", "config": "gpt-kor.toml",
 "container": "auto", "mode": "pre", "overwrite": false, "save_text": false}
```
`container: "auto"`면 빈 컨테이너에 자동 배치. 전체 완료된 작품은 409로 거부(이어하기는 자동 허용).

### `POST /api/jobs/bulk` — 하위 폴더 일괄 추가

```json
{"folder": "20261003_ver1", "config": "gpt-kor.toml", "container": "auto", "mode": "pre"}
```
응답에 `added` / `skipped[{folder, reason}]`.

### `POST /api/jobs/<ID>/cancel` — 실행 중 작업 취소
### `DELETE /api/jobs/<ID>` — 작업 삭제 (실행 중이면 409)

## 예시

```bash
# 진행 요약
curl -s http://localhost:8787/api/summary | jq

# 특정 작업 로그 스트리밍 (2초 폴링)
last=0; while true; do
  r=$(curl -s "http://localhost:8787/api/jobs?id=abc12345&since=$last")
  echo "$r" | jq -r '.lines[]'; last=$(echo "$r" | jq .total)
  sleep 2
done

# 작품 하나 번역 지시
curl -s -X POST http://localhost:8787/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"folder":"20261003_ver1/[A] B (1)","config":"gpt-kor.toml","container":"auto"}'

# GPU 컨테이너 5개로 확장
curl -s -X POST http://localhost:8787/api/gpu \
  -H 'Content-Type: application/json' \
  -d '{"action":"count","count":5}'
```

## 참고

- 서버가 재시작되면 큐(메모리)는 사라지지만 완성/부분 결과물은 디스크에 남는다 → `./startup.sh --resume <날짜폴더>` 가 `mode=pre`로 일괄 재등록한다 (완료 스킵·부분 이어하기·finish 체이닝 자동).
- 부분 완료 판정은 파일 완전성 검사(WebP/PNG/JPEG 트레일러) 포함 — 쓰다 만 페이지는 미완료로 간주해 재처리된다.
- 스택 기동/종료: `./startup.sh` / `./shutdown.sh` (finish 컨테이너 청소 포함).

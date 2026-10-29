#!/usr/bin/env bash
# 만화 번역 스택 기동 — GPU 컨테이너(gpu1~5, pre 레인) + 대시보드/API 서버.
# LLM/렌더링(finish 레인)은 별도의 일회용 CPU 컨테이너로 GUI가 필요할 때마다 docker run
# 하므로 여기서 올릴 컨테이너가 없다 — pre 완료 권은 자동으로 finish에 체이닝된다.
#
# 사용:
#   ./startup.sh                              # 인프라만 기동 (컨테이너 + 대시보드)
#   ./startup.sh --resume 20261003 20261003_ver1   # 기동 후 대기 작업 이어하기 (pre 모드 —
#                                              완료 권 스킵 · 부분 권 이어서 · finish 자동 체이닝)
#
# LLM 레인 처리량 조절 (선택):
#   FINISH_CONCURRENCY=5 FINISH_API_CONCURRENCY=4 ./startup.sh
#     - FINISH_CONCURRENCY:      finish 컨테이너 동시 실행 수 (기본 3)
#     - FINISH_API_CONCURRENCY:  컨테이너당 GLM 동시 요청 수 (기본 3)
#
# 이미 켜져 있으면 아무것도 건드리지 않음 (실행 중 큐 보호) — 언제든 다시 실행해도 안전.

set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/3 Docker 컨테이너 기동 (gpu1~5 정의, 오토스케일러가 목표 개수만 유지) =="
mkdir -p result tmp   # 컨테이너 마운트용 — 없으면 docker가 root 소유로 새로 만들어버림
# 단일 파일 바인드 소스 사전검사 — 파일이 없으면 docker가 컨테이너 안에 디렉터리를 만들어 오염시킴
for f in manga_translator/manga_translator.py \
         manga_translator/mode/local.py \
         manga_translator/mode/finish.py \
         manga_translator/args.py \
         manga_translator/translators/config_gpt.py; do
    if [ ! -f "$f" ]; then
        echo "   ✗ 마운트 소스 파일 없음: $f — compose up 중단" >&2
        exit 1
    fi
done
docker compose --env-file .env -f demo/doc/docker-compose-web-with-gpu.yml up -d
# GPU 접근 단절 점검 (WSL2에서 장기 구동 컨테이너에 발생) — 재생성하면 대부분 복구됨
for c in $(docker ps --filter name=manga_image_translator --format '{{.Names}}'); do
    if ! docker exec "$c" nvidia-smi >/dev/null 2>&1; then
        echo "   ⚠ $c GPU 접근 불가! → docker compose --env-file .env -f demo/doc/docker-compose-web-with-gpu.yml up -d --force-recreate 로 재생성 권장"
    fi
done
docker ps --filter name=manga_image_translator --format '   {{.Names}}  →  {{.Status}}'

echo "== 2/3 대시보드 / API 서버 =="
if curl -sf -o /dev/null http://127.0.0.1:8787/api/summary; then
    echo "   이미 실행 중 — 기존 큐 유지"
else
    # 이전 비정상 종료로 남은 finish 컨테이너(docker run, compose 밖) 청소
    n=$(docker ps -q --filter name=mit-finish | wc -l)
    if [ "$n" -gt 0 ]; then
        docker ps -q --filter name=mit-finish | xargs -r docker rm -f >/dev/null
        echo "   잔여 finish 컨테이너 $n개 정리"
    fi
    # finish 레인 동시성 + GPU 목표 개수 설정 전달 (우선순위: 프로세스 env → .env → 기본값.
    # GPU_CONTAINER_COUNT는 대시보드에서 바꾸면 .gpu_count 파일에 저장되어 여기 값보다 우선한다)
    export FINISH_CONCURRENCY="${FINISH_CONCURRENCY:-3}"
    export FINISH_API_CONCURRENCY="${FINISH_API_CONCURRENCY:-3}"
    export GPU_CONTAINER_COUNT="${GPU_CONTAINER_COUNT:-3}"
    echo "   finish 레인: 컨테이너 ${FINISH_CONCURRENCY}개 × API 동시성 ${FINISH_API_CONCURRENCY} · GPU 목표 ${GPU_CONTAINER_COUNT}개"
    nohup python3 translate_gui.py > /tmp/translate_gui.log 2>&1 &
    for i in $(seq 1 10); do
        curl -sf -o /dev/null http://127.0.0.1:8787/api/summary && break
        sleep 1
    done
    if curl -sf -o /dev/null http://127.0.0.1:8787/api/summary; then
        echo "   시작 완료: http://localhost:8787"
    else
        echo "   실패 — /tmp/translate_gui.log 확인" >&2
        exit 1
    fi
fi

echo "== 3/3 대기 작업 이어하기 (선택) =="
if [ "${1:-}" = "--resume" ] && [ $# -ge 2 ]; then
    shift
    for folder in "$@"; do
        echo "   data/$folder 스캔 중... (pre 모드 — 완료 권 스킵, finish 자동 체이닝)"
        curl -s -X POST http://127.0.0.1:8787/api/jobs/bulk \
            -H 'Content-Type: application/json' \
            -d "{\"folder\":\"$folder\",\"config\":\"gpt-kor.toml\",\"container\":\"auto\",\"mode\":\"pre\"}" \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'   → {d.get(\"added\",0)}권 추가 / {len(d.get(\"skipped\",[]))}권 스킵')" 2>/dev/null \
        || echo "   → $folder 추가 실패"
    done
else
    echo "   건너뜀 — 이어하려면: ./startup.sh --resume 20261003 20261003_ver1"
fi

echo
echo "완료. 대시보드: http://localhost:8787 | API: http://localhost:8787/api/summary"

#!/bin/sh
# 배타 모드 수리 패스 — 로컬 LLM을 한 번에 한 모델만 운용한다:
#   소형 정지 → 대형 기동(프로파일 escalation) → 헬스 대기 → 스플 처리 → 대형 정지 → 소형 재기동
# 실패해도 소형은 반드시 재기동한다(기본 서비스). 사용: ./hymt2_repair.sh [--dry]
set -u
cd "$(dirname "$0")"
COMPOSE=demo/doc/docker-compose-hymt2.yml
IMAGE=zyddnys/manga-image-translator:main

echo '[1/5] 소형 정지'
docker compose -f "$COMPOSE" stop hymt2-small

echo '[2/5] 대형 기동 (첫 회는 다운로드로 수 분~수십 분)'
docker compose -f "$COMPOSE" --profile escalation up -d hymt2-big

echo '[3/5] 대형 헬스 대기'
i=0
until curl -sf -m 2 http://localhost:8012/health >/dev/null 2>&1; do
  i=$((i+1))
  if [ "$i" -gt 240 ]; then
    echo '대형 헬스 타임아웃(20분) — 소형만 복구하고 종료'
    docker compose -f "$COMPOSE" --profile escalation stop hymt2-big
    docker compose -f "$COMPOSE" up -d hymt2-small
    exit 1
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' mit_hymt2_big 2>/dev/null)" != "true" ]; then
    echo '대형 컨테이너 사망 — 로그 확인 후 재시도할 것'
    docker logs mit_hymt2_big 2>&1 | tail -15
    docker compose -f "$COMPOSE" up -d hymt2-small
    exit 1
  fi
  sleep 5
done

echo '[4/5] 스플 수리'
docker run --rm --network hymt2 --entrypoint python -v "$PWD:/app" -w /app \
  -e HYMT2_SMALL_BASE= -e HYMT2_VEC_DIR= -e HYMT2_BIG_BASE=http://mit_hymt2_big:8012/v1 \
  "$IMAGE" -m manga_translator.translators.hymt2_repair "$@"
rc=$?

echo '[5/5] 대형 정지 → 소형 재기동'
docker compose -f "$COMPOSE" --profile escalation stop hymt2-big
docker compose -f "$COMPOSE" up -d hymt2-small
exit $rc

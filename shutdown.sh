#!/usr/bin/env bash
# 만화 번역 스택 종료 — 실행 중 작업 취소 → 대시보드/API → 컨테이너 순으로 안전하게 내림.
#
# 실행 중이던 작업은 완료된 페이지까지만 디스크에 저장되고 중단되며,
# 재기동 시 ./startup.sh --resume <폴더...> 로 끊긴 페이지부터 자동 이어서 진행됨.

set -uo pipefail
cd "$(dirname "$0")"

echo "== 1/3 실행 중 작업 취소 (완료분은 디스크에 유지) =="
if curl -sf -o /dev/null http://127.0.0.1:8787/api/summary; then
    ids=$(curl -s http://127.0.0.1:8787/api/state | python3 -c "
import json,sys
for j in json.load(sys.stdin)['jobs']:
    if j['status'] == 'running':
        print(j['id'])" 2>/dev/null)
    if [ -n "${ids:-}" ]; then
        for jid in $ids; do
            curl -s -X POST "http://127.0.0.1:8787/api/jobs/$jid/cancel" >/dev/null \
                && echo "   취소 요청: $jid"
        done
        echo "   진행 중 페이지 저장 대기..."
        sleep 5
    else
        echo "   실행 중인 작업 없음"
    fi
else
    echo "   대시보드 미실행 — 건너뜀"
fi

echo "== 2/3 대시보드 / API 서버 종료 =="
if pkill -f '[t]ranslate_gui.py' 2>/dev/null; then
    echo "   종료됨 (큐는 사라지지만 완성/부분 결과물은 디스크에 남음)"
else
    echo "   이미 꺼져 있음"
fi

echo "== 3/3 Docker 컨테이너 정지 (gpu1~5 + finish 잔여 컨테이너) =="
docker compose --env-file .env -f demo/doc/docker-compose-web-with-gpu.yml stop
# finish 레인은 docker run(compose 밖)이라 별도 청소 — GUI가 이미 취소했지만 못 받은 고아 방지
n=$(docker ps -q --filter name=mit-finish | wc -l)
if [ "$n" -gt 0 ]; then
    docker ps -q --filter name=mit-finish | xargs -r docker rm -f >/dev/null
    echo "   finish 컨테이너 $n개 정리"
fi
docker ps -a --filter name=manga_image_translator --format '   {{.Names}}  →  {{.Status}}'

echo
echo "완료. 재기동: ./startup.sh  (이어하기: ./startup.sh --resume 20261003 20261003_ver1)"

#!/bin/bash
# 번역 보강 1작업 오케스트레이터 — 원본(일본어) 없이 현재 번역본을 현 품질로 재구성.
#   1단계(GPU 컨테이너): 번역본 이미지에 pre(검출·인페인트·캐시) — 한국어 텍스트도
#      형태 기반 검출기로 영역 확보. 캐시: tmp/aug_pre/<rel> (pkl src=번역본 파일명)
#   2단계(hymt2 네트워크 컨테이너): finish --from-cache --aug-log — 캐시 영역 텍스트를
#      세그먼트 로그의 일본어 원문으로 치환해 재번역·재렌더. 출력=동일 파일명 덮어쓰기.
# 사용: ./augment_runner.sh <gpu컨테이너> <rel>
set -euo pipefail
C=$1; REL=$2
cd "$(dirname "$0")"
IMAGE=zyddnys/manga-image-translator:main

docker exec "$C" python -m manga_translator local \
    -i "translated/$REL" \
    --dest "tmp/aug_pre/$REL" \
    --config-file /app/configs/pass_a.toml \
    --font-path /app/fonts/SUIT-Medium.ttf \
    -f png --preprocess-dump --batch-size 12 --batch-concurrent \
    --models-ttl 300 --use-gpu

# vLLM 온디맨드 — 스케일러 기동 대기 (최대 5분)
for i in $(seq 1 100); do
    curl -sf -m 3 http://localhost:8011/health >/dev/null 2>&1 && break
    sleep 3
done

cmd=(docker run --rm --network hymt2 --user 1000:1000
     --entrypoint python -v "$PWD:/app" -w /app
     -e HYMT2_SMALL_BASE=http://mit_hymt2_small:8011/v1
     -e HYMT2_BIG_BASE=http://mit_hymt2_big:8012/v1)
for host_path in data configs fonts translated tmp result; do
    cmd+=(-v "$PWD/$host_path:/app/$host_path")
done
cmd+=("$IMAGE" -m manga_translator.mode.finish -v
      -i "translated/$REL" --cache "tmp/aug_pre/$REL" -o "translated/$REL"
      --config-file /app/configs/hymt2-kor.toml
      --font-path /app/fonts/SUIT-Medium.ttf --batch-size 12
      --from-cache --overwrite --aug-log "tmp/hymt2-log/$REL.jsonl")
"${cmd[@]}"

"""배타 모드 수리 패스 — 스플에 쌓인 세그먼트를 30B로 일괄 처리해 캐시에 채운다.

로컬 LLM은 한 번에 한 모델만 운용(기본 정책)하므로, 번역 중 즉시 상향 대신
hymt2.py 가 스플 파일(tmp/hymt2-spill/pending.jsonl)에 적립해 둔다. 이 모듈은
그 스플을 소비한다: 대형 서버(HYMT2_BIG_BASE)에 각 쿼리를 보내 성공건을
캐시(tmp/hymt2-cache/big.jsonl)에 추가하고, 실패건만 스플에 되돌려 둔다.
이후 번역에서는 캐시가 Tier 0.6으로 우선 재사용된다.

스왑 오케스트레이션(소형 정지→대형 기동→수리→대형 정지→소형 재기동)은
루트의 hymt2_repair.sh 가 담당한다. 이 모듈은 순수하게 스플 처리만 한다.

사용: python -m manga_translator.translators.hymt2_repair [--dry]
환경: HYMT2_BIG_BASE (필수), HYMT2_SPILL/HYMT2_CACHE (기본값 hymt2.py 와 동일)
"""
import argparse
import asyncio
import json
import os
import sys
from typing import Dict, List

from .hymt2 import HyMT2Translator


def load_spill(path: str) -> List[Dict]:
    """스필 파일 읽기 — norm(q) 기준 중복 제거(첫 등장 우선, refs 보존)."""
    if not os.path.isfile(path):
        return []
    seen = set()
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            q = row.get('q')
            if not q:
                continue
            k = HyMT2Translator._norm(q)
            if k in seen:
                continue
            seen.add(k)
            rows.append(row)
    return rows


def write_cache_append(path: str, results: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


async def run(dry: bool = False) -> Dict[str, int]:
    spill_path = os.environ.get('HYMT2_SPILL', 'tmp/hymt2-spill/pending.jsonl')
    cache_path = os.environ.get('HYMT2_CACHE', 'tmp/hymt2-cache/big.jsonl')
    rows = load_spill(spill_path)
    print(f'스필: {len(rows)}건 (중복 제거 후)')

    # 이미 캐시에 있는 쿼리는 스킵 — 재수리 시 중복 요청 방지
    done = set(HyMT2Translator._norm(r['q']) for r in load_spill(cache_path)
               if r.get('trans'))
    todo = [r for r in rows if HyMT2Translator._norm(r['q']) not in done]
    print(f'미처리: {len(todo)}건 (캐시 스킵 {len(rows) - len(todo)}건)')
    if dry or not todo:
        return {'spill': len(rows), 'todo': len(todo), 'ok': 0, 'fail': 0}
    if not os.environ.get('HYMT2_BIG_BASE'):
        print('HYMT2_BIG_BASE 미설정 — 대형 서버를 지정할 것', file=sys.stderr)
        raise SystemExit(2)

    t = HyMT2Translator()
    if not t._big_base:
        print('대형 클라이언트 초기화 불가', file=sys.stderr)
        raise SystemExit(2)

    async def one(row):
        refs = [(k, v) for k, v in (row.get('refs') or [])]
        trans = await t._big('Korean', row['q'], refs)
        return {'q': row['q'], 'trans': trans} if trans else None

    outs = await asyncio.gather(*(one(r) for r in todo))
    ok = [o for o in outs if o]
    fail = [r for r, o in zip(todo, outs) if not o]
    write_cache_append(cache_path, ok)
    # 실패건만 스플에 되돌린다 — 다음 수리 패스가 재시도
    os.makedirs(os.path.dirname(spill_path) or '.', exist_ok=True)
    with open(spill_path, 'w', encoding='utf-8') as f:
        for r in fail:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'수리 완료: 성공 {len(ok)} → 캐시 추가, 실패 {len(fail)} → 스플 유지')
    return {'spill': len(rows), 'todo': len(todo), 'ok': len(ok), 'fail': len(fail)}


def main():
    ap = argparse.ArgumentParser(prog='manga_translator.translators.hymt2_repair')
    ap.add_argument('--dry', action='store_true', help='처리 없이 건수만 출력')
    args = ap.parse_args()
    asyncio.run(run(dry=args.dry))


if __name__ == '__main__':
    main()

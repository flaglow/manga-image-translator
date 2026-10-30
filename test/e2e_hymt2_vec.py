# 벡터 티어 e2e — 실제 df.flaglow.cc 임베딩 + configs/hymt2-vec DB (vLLM 없이 검증).
# 사용: docker run --rm --entrypoint python -v "$PWD:/app" -w /app \
#        -e HYMT2_GLOSSARY= -e HYMT2_BIG_BASE= -e HYMT2_SMALL_BASE=http://localhost:9/v1 \
#        zyddnys/manga-image-translator:main test/e2e_hymt2_vec.py
import asyncio
import logging
import os

os.environ.setdefault('HYMT2_TM', '/app/configs/hymt2-tm.tsv')
os.environ.setdefault('HYMT2_GLOSSARY', '')
os.environ.setdefault('HYMT2_BIG_BASE', '')
os.environ.setdefault('HYMT2_SMALL_BASE', 'http://localhost:9/v1')
os.environ.setdefault('HYMT2_VEC_DIR', '/app/configs/hymt2-vec')

from manga_translator.utils import init_logging, set_log_level
init_logging()
set_log_level(logging.INFO)

from manga_translator.translators import dispatch, TranslatorChain


async def main():
    chain = TranslatorChain('hymt2:KOR')
    # TM히트 / TM미스→벡터근접(참조 or 재사용 후 small 다운→빈칸) / 일반문장
    queries = ['うん', 'なんでだろうな', '今日はいい天気ですね']
    print('RESULT:', await dispatch(chain, queries))


asyncio.run(main())

# 등록→dispatch→번역기 초기화→TM 티어까지 HTTP 없이 검증 (vLLM 미기동 상태용).
# 사용: docker run --rm --entrypoint python -v "$PWD:/app" -w /app \
#        -e HYMT2_GLOSSARY= -e HYMT2_BIG_BASE= zyddnys/manga-image-translator:main test/e2e_hymt2_tm.py
import asyncio
import os

os.environ.setdefault('HYMT2_TM', '/app/configs/hymt2-tm.tsv')
os.environ.setdefault('HYMT2_GLOSSARY', '')
os.environ.setdefault('HYMT2_BIG_BASE', '')

from manga_translator.translators import dispatch, TranslatorChain


async def main():
    chain = TranslatorChain('hymt2:KOR')
    # TM히트(share 0.99) / 괄호+행끝부호 정규화 TM히트(share 0.76) / 비대상 통과
    queries = ['うん', '「んっ！」', '…']
    print('RESULT:', await dispatch(chain, queries))


asyncio.run(main())

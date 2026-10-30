"""HyMT2Translator 단위 테스트 — vLLM HTTP 호출은 전부 mock.

실행(로컬 .venv는 깨져 있음 — 파이프라인 이미지 사용):
  docker run --rm --entrypoint bash -v "$PWD:/app" -w /app \
    zyddnys/manga-image-translator:main -c "python -m pytest test/test_hymt2.py -q"
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import os


def _fake_resp(content: str):
    m = MagicMock()
    m.choices = [MagicMock()]
    m.choices[0].message.content = content
    return m


@pytest.fixture
def make_translator(tmp_path, monkeypatch):
    def _make(tm_lines=(), glossary_lines=(), big=True, glm=False):
        tm = tmp_path / 'tm.tsv'
        tm.write_text('\n'.join(tm_lines) + '\n', encoding='utf-8')
        gl = tmp_path / 'glossary.tsv'
        gl.write_text('\n'.join(glossary_lines) + '\n', encoding='utf-8')
        monkeypatch.setenv('HYMT2_TM', str(tm))
        monkeypatch.setenv('HYMT2_GLOSSARY', str(gl) if glossary_lines else '')
        monkeypatch.setenv('HYMT2_SMALL_BASE', 'http://fake-small:1/v1')
        monkeypatch.setenv('HYMT2_BIG_BASE', 'http://fake-big:1/v1' if big else '')
        monkeypatch.setenv('HYMT2_GL_FALLBACK', '1' if glm else '0')
        monkeypatch.setenv('HYMT2_CONCURRENCY', '4')
        from manga_translator.translators.hymt2 import HyMT2Translator
        return HyMT2Translator()
    return _make


@pytest.mark.asyncio
async def test_tm_hit_skips_llm(make_translator):
    t = make_translator(tm_lines=['あっ\t앗'])
    t._client_small.chat.completions.create = AsyncMock()
    out = await t._translate('auto', 'Korean', ['あっ'])
    assert out == ['앗']
    assert t._client_small.chat.completions.create.await_count == 0
    assert t._stat['tm'] == 1


@pytest.mark.asyncio
async def test_tm_trailing_punct_fallback(make_translator):
    # '「あっ！」' → norm 'あっ!' → 1차 미스 → 행끝 부호 제거 2차 히트
    t = make_translator(tm_lines=['あっ\t앗'])
    t._client_small.chat.completions.create = AsyncMock()
    out = await t._translate('auto', 'Korean', ['「あっ！」'])
    assert out == ['앗']
    assert t._client_small.chat.completions.create.await_count == 0


@pytest.mark.asyncio
async def test_small_pass(make_translator):
    t = make_translator()
    t._client_small.chat.completions.create = AsyncMock(return_value=_fake_resp('무슨 일이야?'))
    out = await t._translate('auto', 'Korean', ['どうしたの？'])
    assert out == ['무슨 일이야?']
    assert t._stat['small'] == 1 and t._stat['big'] == 0


@pytest.mark.asyncio
async def test_small_gate_fail_then_retry_ok(make_translator):
    # 1차: 원문 통과(한글 없음) → 게이트 실패, 힌트 재시도에서 통과
    t = make_translator()
    t._client_small.chat.completions.create = AsyncMock(
        side_effect=[_fake_resp('どうしたの？'), _fake_resp('무슨 일이야?')])
    out = await t._translate('auto', 'Korean', ['どうしたの？'])
    assert out == ['무슨 일이야?']
    assert t._stat['small'] == 1 and t._stat['small_retry_ok'] == 1


@pytest.mark.asyncio
async def test_escalate_to_big(make_translator):
    # small 2회 모두 실패 → big이 처리
    t = make_translator()
    t._client_small.chat.completions.create = AsyncMock(
        side_effect=[_fake_resp('どうした'), _fake_resp('どうした')])
    t._client_big.chat.completions.create = AsyncMock(return_value=_fake_resp('어떡해'))
    out = await t._translate('auto', 'Korean', ['どうした'])
    assert out == ['어떡해']
    assert t._stat['big'] == 1 and t._stat['small'] == 0


@pytest.mark.asyncio
async def test_overlength_falls_back(make_translator):
    # 과길이 2배 이상 → 재시도도 과길이 → big
    long_kr = '이것은 아주 많이 길어진 한국어 번역문입니다.' * 4
    t = make_translator()
    t._client_small.chat.completions.create = AsyncMock(
        side_effect=[_fake_resp(long_kr), _fake_resp(long_kr)])
    t._client_big.chat.completions.create = AsyncMock(return_value=_fake_resp('짧게'))
    out = await t._translate('auto', 'Korean', ['短く'])
    assert out == ['짧게']
    assert t._stat['big'] == 1


@pytest.mark.asyncio
async def test_big_disabled_empty_result(make_translator):
    # big 끔(HYMT2_BIG_BASE='') + small 실패 → 빈 칸
    t = make_translator(big=False)
    t._client_small.chat.completions.create = AsyncMock(
        side_effect=[Exception('conn'), Exception('conn'), Exception('conn'), Exception('conn')])
    out = await t._translate('auto', 'Korean', ['どうした'])
    assert out == ['']
    assert t._stat['empty'] == 1


@pytest.mark.asyncio
async def test_symbol_only_kept(make_translator):
    # 번역 불가 세그먼트(부호만) — small 실패해도 원문 유지
    t = make_translator()
    fail = AsyncMock(side_effect=Exception('x'))
    t._client_small.chat.completions.create = fail
    t._client_big.chat.completions.create = fail
    out = await t._translate('auto', 'Korean', ['…!'])
    assert out == ['…!']


@pytest.mark.asyncio
async def test_connection_error_escalates(make_translator):
    # small 서버 다운 → 전부 big으로
    t = make_translator()
    t._client_small.chat.completions.create = AsyncMock(side_effect=Exception('refused'))
    t._client_big.chat.completions.create = AsyncMock(return_value=_fake_resp('좋아'))
    out = await t._translate('auto', 'Korean', ['いいよ'])
    assert out == ['좋아']
    assert t._stat['big'] == 1


def test_norm_variants():
    from manga_translator.translators.hymt2 import HyMT2Translator
    n = HyMT2Translator._norm
    assert n('あっ ！') == n('あっ!')
    assert n('はぁ…') == n('はぁ...')
    assert n('「はい」') == n('はい')


def test_prompt_official_shape(make_translator):
    t = make_translator(glossary_lines=['先輩\t선배', 'ギルド\t길드', '無関係\t무관'])
    p = t._build_prompt('Korean', '先輩、ギルドへ行こう', '')
    assert p.startswith('Reference the following translations:')
    assert '先輩 => 선배' in p and 'ギルド => 길드' in p
    assert '無関係' not in p  # 세그먼트와 무관한 용어는 미주입
    assert 'Translate the following text into Korean' in p
    assert p.endswith('先輩、ギルドへ行こう')


def test_glossary_cap(make_translator):
    many = [f'語{i}\t역{i}' for i in range(20)] + ['本気\t진심']
    t = make_translator(glossary_lines=many)
    p = t._build_prompt('Korean', '本気' + ''.join(f'語{i}' for i in range(20)), '')
    assert p.count(' => ') <= t._GLOSSARY_MAX


def test_gate_rules():
    from manga_translator.translators.hymt2 import HyMT2Translator as H
    assert H._gate_ok('どうした', '어떡해')
    assert not H._gate_ok('どうした', 'どうした')      # 원문 통과
    assert not H._gate_ok('どうした', '')               # 빈 출력
    assert not H._gate_ok('短く', '너무나도 길게 늘어진 번역문이 되어버렸다')  # 과길이
    assert H._gate_ok('…!', '')                        # 부호뿉 원문 — 통과
    assert H._gate_ok('2024', '2024')                  # 숫자 유지

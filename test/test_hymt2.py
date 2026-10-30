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
    def _make(tm_lines=(), glossary_lines=(), big=True, glm=False, vec=None,
              emb_model='test-emb', build_model=None, exclusive=False, cache_lines=()):
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
        # 기본은 즉시 상향(기존 테스트 동작 유지) — 배타 테스트는 exclusive=True
        monkeypatch.setenv('HYMT2_EXCLUSIVE', '1' if exclusive else '0')
        monkeypatch.setenv('HYMT2_CACHE', str(tmp_path / 'cache.jsonl'))
        monkeypatch.setenv('HYMT2_SPILL', str(tmp_path / 'spill.jsonl'))
        if cache_lines:
            (tmp_path / 'cache.jsonl').write_text(
                '\n'.join(cache_lines) + '\n', encoding='utf-8')
        monkeypatch.setenv('HYMT2_EMB_BASE', 'http://fake-emb:1/v1')
        monkeypatch.setenv('HYMT2_EMB_MODEL', emb_model)
        if vec is not None:
            import json as _json

            import numpy as np
            vd = tmp_path / 'vec'
            vd.mkdir(exist_ok=True)
            keys, trans, shares, vecs = zip(*vec)
            emb = np.array(vecs, dtype=np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
            np.save(vd / 'emb.npy', emb.astype(np.float16))
            with open(vd / 'meta.tsv', 'w', encoding='utf-8') as f:
                for k, t, s in zip(keys, trans, shares):
                    f.write(f'{k}\t{t}\t9\t{s}\n')
            with open(vd / 'BUILD.json', 'w', encoding='utf-8') as f:
                _json.dump({'model': build_model or emb_model, 'dims': 3, 'entries': len(keys)}, f)
            monkeypatch.setenv('HYMT2_VEC_DIR', str(vd))
        else:
            monkeypatch.setenv('HYMT2_VEC_DIR', '')
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


VEC_DB = [  # key, trans, share, vector (3차원 — 직교 기저로 유사도 설계)
    ('あ', '아', 0.8, (1.0, 0.0, 0.0)),
    ('い', '이', 0.6, (0.0, 1.0, 0.0)),
    ('う', '우', 0.9, (0.0, 0.0, 1.0)),
]


def _fake_embed(vectors):
    async def _embed(self, texts):
        import numpy as np
        emb = np.array(vectors, dtype=np.float32)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return _embed


@pytest.mark.asyncio
async def test_exclusive_spills_instead_of_big(make_translator):
    # 배타 모드(기본): 소형 실패 시 즉시 상향 없이 스플 적립 — 수리 패스가 처리
    t = make_translator(exclusive=True)
    t._client_small.chat.completions.create = AsyncMock(
        side_effect=[_fake_resp('どうした'), _fake_resp('どうした')])
    t._client_big.chat.completions.create = AsyncMock(return_value=_fake_resp('어떡해'))
    out = await t._translate('auto', 'Korean', ['どうした'])
    assert out == [''] and t._stat['spill'] == 1 and t._stat['big'] == 0
    t._client_big.chat.completions.create.assert_not_awaited()
    import json as _json
    rows = [_json.loads(l) for l in open(t._spill_path, encoding='utf-8') if l.strip()]
    assert rows[0]['q'] == 'どうした'


@pytest.mark.asyncio
async def test_cache_tier_hit(make_translator):
    # 수리 캐시에 있는 원문은 LLM 호출 없이 재사용 (Tier 0.6)
    t = make_translator(exclusive=True, cache_lines=['{"q": "行こう", "trans": "가자"}'])
    t._client_small.chat.completions.create = AsyncMock(return_value=_fake_resp('x'))
    out = await t._translate('auto', 'Korean', ['行こう'])
    assert out == ['가자'] and t._stat['cache'] == 1
    t._client_small.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_repair_run(make_translator, monkeypatch):
    # 스플 2건 → 대형 1건 성공(캐시 추가)·1건 실패(스필 유지)
    t0 = make_translator(exclusive=True)
    with open(t0._spill_path, 'w', encoding='utf-8') as f:
        f.write('{"q": "行こう", "refs": []}\n{"q": "来るぞ", "refs": [["行こう", "가자"]]}\n')
    from manga_translator.translators import hymt2_repair
    from manga_translator.translators.hymt2 import HyMT2Translator as H

    async def fake_big(self, to_lang, query, refs=None):
        return '간다' if query == '来るぞ' else None

    monkeypatch.setattr(H, '_big', fake_big)
    st = await hymt2_repair.run()
    assert st == {'spill': 2, 'todo': 2, 'ok': 1, 'fail': 1}
    import json as _json
    cache_rows = [_json.loads(l) for l in open(t0._cache_path, encoding='utf-8') if l.strip()]
    assert cache_rows[-1]['trans'] == '간다'
    spill_rows = [_json.loads(l) for l in open(t0._spill_path, encoding='utf-8') if l.strip()]
    assert [r['q'] for r in spill_rows] == ['行こう']


@pytest.mark.asyncio
async def test_vec_reuse_and_ref(make_translator, monkeypatch):
    t = make_translator(vec=VEC_DB)
    # 쿼리: 'ぁ'→e1과 동일(sim 1.0, share 0.8) 재사용 / 'ぃ'→e2 0.95(share 0.6) 참조 /
    #       'ぇ'→저유사도 미참조. DB 키와 글자를 달리해 동일키 제외 로직 회피
    monkeypatch.setattr(type(t), '_embed_batch', _fake_embed([(1, 0, 0), (0, 0.95, 0.312), (0.6, 0.6, 0.5)]))
    prompts = []

    async def fake_create(**kwargs):
        prompts.append(kwargs['messages'][0]['content'])
        return _fake_resp('결과')

    t._client_small.chat.completions.create = AsyncMock(side_effect=fake_create)
    out = await t._translate('auto', 'Korean', ['ぁ', 'ぃ', 'ぇ'])
    assert out[0] == '아'                      # 벡터 재사용 — LLM 호출 없음
    assert t._stat['vec_reuse'] == 1 and t._stat['vec_ref'] == 1
    assert t._client_small.chat.completions.create.await_count == 2
    assert any('い => 이' in p for p in prompts)   # 참조 주입 확인
    assert not any('う => 우' in p for p in prompts)


@pytest.mark.asyncio
async def test_vec_skip_on_api_error(make_translator, monkeypatch):
    t = make_translator(vec=VEC_DB)

    async def boom(self, texts):
        raise RuntimeError('embed down')

    monkeypatch.setattr(type(t), '_embed_batch', boom)
    t._client_small.chat.completions.create = AsyncMock(return_value=_fake_resp('좋아'))
    out = await t._translate('auto', 'Korean', ['ぁ'])
    assert out == ['좋아'] and t._stat['small'] == 1


@pytest.mark.asyncio
async def test_vec_model_mismatch_disables(make_translator):
    # DB 빌드 모델(other-model) ≠ 요청 모델(test-emb) → 티어 비활성
    t = make_translator(vec=VEC_DB, build_model='other-model')
    assert t._vec is None
    t._client_small.chat.completions.create = AsyncMock(return_value=_fake_resp('좋아'))
    out = await t._translate('auto', 'Korean', ['ぁ'])
    assert out == ['좋아'] and t._stat['small'] == 1


@pytest.mark.asyncio
async def test_vec_search_excludes_same_key(make_translator, monkeypatch):
    # 쿼리 키가 DB 행 키와 동일하면(TM이 이미 캐치) 다음 근접행을 반환
    t = make_translator(vec=VEC_DB)
    monkeypatch.setattr(type(t), '_embed_batch', _fake_embed([(1, 0, 0)]))
    hits = await t._vec_search(['あ'])
    sim, row = hits[0]
    assert t._vec['keys'][row] != 'あ' and sim < 0.5


def test_gate_rules():
    from manga_translator.translators.hymt2 import HyMT2Translator as H
    assert H._gate_ok('どうした', '어떡해')
    assert not H._gate_ok('どうした', 'どうした')      # 원문 통과
    assert not H._gate_ok('どうした', '')               # 빈 출력
    assert not H._gate_ok('短く', '너무나도 길게 늘어진 번역문이 되어버렸다')  # 과길이
    assert H._gate_ok('…!', '')                        # 부호뿉 원문 — 통과
    assert H._gate_ok('2024', '2024')                  # 숫자 유지

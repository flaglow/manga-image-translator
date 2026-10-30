import asyncio
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import CommonTranslator

try:
    import openai
except ImportError:
    openai = None


class HyMT2Translator(CommonTranslator):
    """Hy-MT2 로컬 LLM 번역기 (vLLM OpenAI 호환 서버 2단).

    티어 구조 (docs/로컬 LLM 번역 파이프라인 설계__hymt2.md 참조):
      Tier 0   TM 정확일치 (configs/hymt2-tm.tsv — 프로덕션 GLM 코퍼스 다수역)
      Tier 0.5 벡터 근접검색 (configs/hymt2-vec/ — df.flaglow.cc 임베딩 + numpy 평탄 인덱스)
               고유사도·고품질행은 재사용, 그 아래는 공식 용어집 형식으로 참조 주입
      Tier 1   Hy-MT2-1.8B-FP8 전수 1차 (세그먼트별 독립 요청, 동시성 세마포어)
      게이트   빈 출력/한글 부재/원문 2배 과길이/원문 통과 → 힌트 1회 재시도
      Tier 2   Hy-MT2-30B-A3B-FP8 실패분 일괄 (big 서버 설정 시)
               → (옵션) GLM 비상 폴백 → 빈 칸

    목표 지표 "% resolved by ≤1.8B" 를 페이지 요약 로그로 상시 추적한다.
    프롬프트/샘플링은 Hy-MT2 공식값 고정(모델 전용 설정 우선):
      1.8B t0.7/p0.6/k20/rep1.05 · 30B t0.7/p1.0/rep1.0 · 시스템 프롬프트 없음.

    배포: demo/doc/docker-compose-hymt2.yml (포트 8011/8012, big은 profile escalation).
    선택: configs/hymt2-kor.toml. 재정의: HYMT2_* 환경변수.
    """
    _LANGUAGE_CODE_MAP = {'JPN': 'Japanese', 'ENG': 'English', 'KOR': 'Korean'}

    _TIMEOUT = 30          # 1.8B 요청 타임아웃(초) — HYMT2_TIMEOUT
    _BIG_TIMEOUT = 90      # 30B 요청 타임아웃(초) — HYMT2_BIG_TIMEOUT
    _CONNECT_ATTEMPTS = 2  # 서버 불응 시 재시도
    _LENGTH_HARD_CAP = 2   # 원문 대비 과길이 배수 (custom_openai 계승)
    _GLOSSARY_MAX = 8      # 세그먼트당 주입 용어 상한

    # 게이트 실패 재시도 시 프롬프트 끝에 붙는 힌트 (1.8B 전용, 최소 지시)
    _HINT_KOREAN = 'Note: the translation result must be natural Korean.'
    _HINT_LENGTH = 'Note: keep the translation concise, not much longer than the source.'

    _HANGUL = re.compile(r'[가-힣]')
    _JP_OR_EN = re.compile(r'[぀-ヿ一-鿿a-zA-Z]')
    _TRAIL_PUNCT = re.compile(r'[?!.…~♥]+$')  # 행끝 부호 — TM 2차 조회에서만 제거

    def __init__(self):
        super().__init__()
        if openai is None:
            raise RuntimeError('openai 패키지가 없습니다 — 파이프라인 이미지 확인 필요')
        self._small_base = os.environ.get('HYMT2_SMALL_BASE', 'http://localhost:8011/v1')
        self._big_base = os.environ.get('HYMT2_BIG_BASE', 'http://localhost:8012/v1')
        self._small_model = os.environ.get('HYMT2_SMALL_MODEL', 'hymt2-1.8b')
        self._big_model = os.environ.get('HYMT2_BIG_MODEL', 'hymt2-30b')
        self._escalate_big = bool(self._big_base) and os.environ.get('HYMT2_ESCALATE', '1') != '0'
        self._glm_fallback = os.environ.get('HYMT2_GL_FALLBACK', '') == '1'
        self._timeout = float(os.environ.get('HYMT2_TIMEOUT', '') or self._TIMEOUT)
        self._big_timeout = float(os.environ.get('HYMT2_BIG_TIMEOUT', '') or self._BIG_TIMEOUT)
        self._sem = asyncio.Semaphore(int(os.environ.get('HYMT2_CONCURRENCY', '16')))
        self._client_small = openai.AsyncOpenAI(
            api_key='local', base_url=self._small_base, timeout=self._timeout)
        self._client_big = openai.AsyncOpenAI(
            api_key='local', base_url=self._big_base, timeout=self._big_timeout)
        self._client_glm = None
        if self._glm_fallback:
            from .keys import (CUSTOM_OPENAI_API_KEY, CUSTOM_OPENAI_API_BASE, CUSTOM_OPENAI_MODEL)
            self._client_glm = openai.AsyncOpenAI(
                api_key=CUSTOM_OPENAI_API_KEY or 'x', base_url=CUSTOM_OPENAI_API_BASE, timeout=60)
            self._glm_model = CUSTOM_OPENAI_MODEL
        self._tm = self._load_tm(os.environ.get('HYMT2_TM', 'configs/hymt2-tm.tsv'))
        self._glossary = self._load_glossary(os.environ.get(
            'HYMT2_GLOSSARY', 'configs/hdtrans-userdict.txt'))

        # 벡터 근접검색 — DB(configs/hymt2-vec/)가 있고 임베딩 API가 설정돼 있을 때만 동작
        self._emb_base = os.environ.get('HYMT2_EMB_BASE', 'https://df.flaglow.cc/embedder/v1')
        self._emb_model = os.environ.get('HYMT2_EMB_MODEL', 'bge-m3-mlx-8bit')
        self._vec_reuse_sim = float(os.environ.get('HYMT2_VEC_REUSE_SIM', '0.985'))
        self._vec_ref_sim = float(os.environ.get('HYMT2_VEC_REF_SIM', '0.88'))
        self._vec = self._load_vec(os.environ.get('HYMT2_VEC_DIR', 'configs/hymt2-vec'))
        self._client_emb = None
        if self._vec is not None and self._emb_base:
            self._client_emb = openai.AsyncOpenAI(
                api_key='local', base_url=self._emb_base, timeout=30)

        # 페이지 요약 로그용 카운터 (translate() 호출당 리셋)
        self._stat = {}
        # 세그먼트별 최종 처리 티어 (평가·디버깅용): 'tm','vec','small','small_retry',
        # 'big','glm','empty','pass' — 참조 주입됐으면 'ref_' 접두
        self._trace = []

    def parse_args(self, args):
        pass  # toml 확장 필드 없음 — env로만 재정의 (hdtrans 관례)

    # ── 리소스 로딩 ──────────────────────────────────────────────

    @staticmethod
    def _norm(t: str) -> str:
        """TM 키 정규화 — hdtrans-poc expand_dict.norm() + 대사 괄호 제거.
        build_tm.py 의 norm() 과 반드시 같은 규칙을 유지할 것."""
        t = re.sub(r'\s+', '', t.strip())
        return (t.replace('！', '!').replace('？', '?').replace('…', '...')
                .replace('〜', '~').replace('♡', '♥').replace('"', '').replace("'", '')
                .replace('、', ',').replace('。', '.').replace('「', '').replace('」', '')
                .replace('『', '').replace('』', '').lower())

    def _load_tm(self, path: str) -> Dict[str, str]:
        """정확일치 TM — norm(원문) → 번역. 헤더 # 는 주석.

        1차 키(전체 norm)와 별도로 행끝 부호를 걷은 2차 인덱스를 함께 만든다
        (OCR 세그먼트의 'あっ！'↔코퍼스 'あっ' 변형 대응). 파일이 count 내림차순이므로
        2차 인덱스는 첫 등장(최빈)이 우선한다.
        """
        if not path or not os.path.isfile(path):
            if path:
                self.logger.warning(f'TM 파일 없음 — TM 없이 진행: {path}')
            return {}
        tm = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                # 주석(# 헤더)은 탭이 없다 — '#'로 시작하는 키(해시태그 대사)는 보존
                line = line.rstrip('\n')
                if not line or '\t' not in line:
                    continue
                cols = line.split('\t')
                if len(cols) < 2:
                    continue
                key, val = cols[0], cols[1]
                tm.setdefault(key, val)
                stripped = self._TRAIL_PUNCT.sub('', key)
                if stripped and stripped != key:
                    tm.setdefault(stripped, val)
        self.logger.info(f'TM 로드: {len(tm)} 키 ({path})')
        return tm

    def _tm_lookup(self, query: str) -> Optional[str]:
        """정확키 → 행끝부호 제거키 순 조회."""
        key = self._norm(query)
        if key in self._tm:
            return self._tm[key]
        stripped = self._TRAIL_PUNCT.sub('', key)
        if stripped and stripped != key:
            return self._tm.get(stripped)
        return None

    def _load_glossary(self, path: str) -> List[Tuple[str, str]]:
        """용어집 TSV(원문<TAB>번역) — hdtrans-userdict.txt 형식 그대로 사용."""
        if not path or not os.path.isfile(path):
            if path:
                self.logger.warning(f'용어집 없음 — 주입 없이 진행: {path}')
            return []
        entries = []
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or '\t' not in line:  # 주석 헤더 = 탭 없는 줄
                    continue
                k, v = line.split('\t', 1)
                if k and v:
                    entries.append((k.strip(), v.strip()))
        self.logger.info(f'용어집 로드: {len(entries)} 엔트리 ({path})')
        return entries

    # ── 벡터 근접검색 (Tier 0.5) ────────────────────────────────

    def _load_vec(self, vec_dir: str):
        """configs/hymt2-vec/{emb.npy, meta.tsv, BUILD.json} — build_vec.py 산출물.

        numpy 평탄 인덱스: 45K×1024 matvec 가 <1ms (faiss 불필요 — 신규 네이티브
        의존성 0, 컨테이너에 numpy 이미 내장). emb.npy(fp16)를 fp32로 정규화해 상주.
        BUILD.json 의 모델명이 HYMT2_EMB_MODEL 과 다르면 쿼리 벡터 공간이 어긋나므로
        티어를 비활성화한다. 디렉터리가 없어도 None (자동 비활성).
        """
        import json
        emb_path = os.path.join(vec_dir, 'emb.npy')
        meta_path = os.path.join(vec_dir, 'meta.tsv')
        build_path = os.path.join(vec_dir, 'BUILD.json')
        if not (os.path.isfile(emb_path) and os.path.isfile(meta_path)):
            return None
        try:
            build = json.load(open(build_path, encoding='utf-8'))
        except Exception:
            build = {}
        if build.get('model') and build.get('model') != self._emb_model:
            self.logger.warning(
                f"벡터 DB 모델 불일치 (DB={build.get('model')} ≠ 요청={self._emb_model}) "
                '— 벡터 티어 비활성. HYMT2_EMB_MODEL 을 DB 빌드 모델로 맞출 것')
            return None
        emb = np.load(emb_path).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / (norms + 1e-9)
        keys, trans, shares = [], [], []
        with open(meta_path, encoding='utf-8') as f:
            for line in f:
                if '\t' not in line:  # 주석 헤더 = 탭 없는 줄 ('#' 키 보존)
                    continue
                cols = line.rstrip('\n').split('\t')
                keys.append(cols[0])
                trans.append(cols[1])
                shares.append(float(cols[3]) if len(cols) > 3 else 0.0)
        if len(keys) != emb.shape[0]:
            self.logger.warning(
                f'벡터 DB 불일치 — meta {len(keys)}행 vs emb {emb.shape[0]}행, 티어 비활성')
            return None
        self.logger.info(f'벡터 TM 로드: {emb.shape[0]}행 × {emb.shape[1]}차원 ({vec_dir})')
        return {'emb': emb, 'keys': keys, 'trans': trans, 'shares': shares}

    async def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """임베딩 API(df.flaglow.cc /embedder/v1, OpenAI 호환)로 정규화 벡터 반환."""
        chunks = [texts[i:i + 64] for i in range(0, len(texts), 64)]

        async def one(chunk):
            r = await self._client_emb.embeddings.create(model=self._emb_model, input=chunk)
            data = sorted(r.data, key=lambda d: d.index)
            return [d.embedding for d in data]

        vectors = [v for part in await asyncio.gather(*[one(c) for c in chunks]) for v in part]
        emb = np.array(vectors, dtype=np.float32)
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    async def _vec_search(self, keys: List[str]) -> List[Optional[Tuple[float, int]]]:
        """각 키의 최근접 행 — 동일 키(=TM이 이미 캐치)는 제외하고 (sim, row) 반환."""
        q = await self._embed_batch(keys)
        sims = q @ self._vec['emb'].T
        out = []
        for i, key in enumerate(keys):
            row_sims = sims[i]
            row = int(np.argmax(row_sims))
            if self._vec['keys'][row] == key:
                row_sims = row_sims.copy()
                row_sims[row] = -1.0
                row = int(np.argmax(row_sims))
            out.append((float(row_sims[row]), row))
        return out

    # ── 게이트 ──────────────────────────────────────────────────

    @classmethod
    def _has_hangul(cls, s: str) -> bool:
        return bool(cls._HANGUL.search(s or ''))

    @classmethod
    def _needs_translation(cls, s: str) -> bool:
        """번역 가능한 문자(가나/한자/알파벳)가 있는지 — 부호만인 세그먼트는 통과 게이트 제외."""
        return bool(cls._JP_OR_EN.search(s or ''))

    @staticmethod
    def _char_len(s: str) -> int:
        return len(re.sub(r'\s+', '', s or ''))

    @classmethod
    def _gate_ok(cls, query: str, trans: str) -> bool:
        """통과 조건: 비었거나 부호뿐인 원문은 그대로 두고, 그 외는 한글 필수."""
        if not trans:
            return not cls._needs_translation(query)
        if cls._needs_translation(query) and not cls._has_hangul(trans):
            return False
        q = cls._char_len(query)
        return not (q >= 2 and cls._char_len(trans) >= cls._LENGTH_HARD_CAP * q)

    # ── 프롬프트 (Hy-MT2 공식형) ─────────────────────────────────

    def _build_prompt(self, to_lang: str, query: str, hint: str = '',
                      refs: Optional[List[Tuple[str, str]]] = None) -> str:
        """공식 프롬프트 + 용어집/벡터참조 병합 (단일 Reference 블록, 상한 8쌍).

        스타일 라인은 Hy-MT2 공식 스타일 지시 형식(【...】 엄격 준수)을 쓴다 —
        평가에서 1.8B가 해요체로 기우는 계통 편차(코퍼스 관례는 반말 대사체) 교정용.
        """
        parts = []
        pairs = [kv for kv in self._glossary if kv[0] in query][:self._GLOSSARY_MAX]
        for r in (refs or []):
            if len(pairs) >= self._GLOSSARY_MAX:
                break
            if r not in pairs:
                pairs.append(r)
        if pairs:
            ref = '\n'.join(f'{k} => {v}' for k, v in pairs)
            parts.append(f'Reference the following translations:\n{ref}')
        parts.append(
            f'The translation style must strictly conform to 【manga dialogue: natural colloquial '
            f'{to_lang}, casual speech, short and punchy】.')
        parts.append(
            f'Translate the following text into {to_lang}, note that you need only output '
            f'the translated result without any additional explanation:')
        parts.append(query)
        if hint:
            parts.append(hint)
        return '\n\n'.join(parts)

    async def _request(self, client, model: str, to_lang: str, query: str,
                       hint: str, top_p: float, extra: dict,
                       refs: Optional[List[Tuple[str, str]]] = None) -> Optional[str]:
        """1회 요청 — 예외/빈응답은 None (호출자가 에스컬레이션)."""
        for attempt in range(self._CONNECT_ATTEMPTS):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{'role': 'user', 'content': self._build_prompt(to_lang, query, hint, refs)}],
                    temperature=0.7,
                    top_p=top_p,
                    max_tokens=512,
                    extra_body=extra,
                )
                content = (r.choices[0].message.content or '').strip()
                # 소형 모델이 프롬프트를 따옴표로 되돌리는 경우 제거
                if len(content) >= 2 and content[0] == content[-1] and content[0] in '"\'':
                    content = content[1:-1].strip()
                return content or None
            except Exception as e:
                if attempt == self._CONNECT_ATTEMPTS - 1:
                    self.logger.warning(f'HyMT2 요청 실패 ({model}): {e}')
                    return None
                await asyncio.sleep(0.5)

    async def _small(self, to_lang: str, query: str,
                     refs: Optional[List[Tuple[str, str]]] = None) -> Optional[str]:
        extra = {'top_k': 20, 'repetition_penalty': 1.05}
        async with self._sem:
            trans = await self._request(
                self._client_small, self._small_model, to_lang, query, '', 0.6, extra, refs)
        if trans is not None and self._gate_ok(query, trans):
            return trans
        # 게이트 실패 — 힌트 1회 재시도
        hint = (self._HINT_KOREAN if (trans is None or not self._has_hangul(trans))
                else self._HINT_LENGTH)
        async with self._sem:
            retry = await self._request(
                self._client_small, self._small_model, to_lang, query, hint, 0.6, extra, refs)
        if retry is not None and self._gate_ok(query, retry):
            self._stat['small_retry_ok'] = self._stat.get('small_retry_ok', 0) + 1
            return retry
        # 과길이 차선: 게이트 밖이어도 원문 이상으로는 안 긴 최단 후보
        for cand in (retry, trans):
            if cand and self._has_hangul(cand) and self._char_len(cand) < 4 * self._char_len(query):
                return cand
        self._stat['small_fail'] = self._stat.get('small_fail', 0) + 1
        return None

    async def _big(self, to_lang: str, query: str,
                   refs: Optional[List[Tuple[str, str]]] = None) -> Optional[str]:
        extra = {'repetition_penalty': 1.0}
        async with self._sem:
            trans = await self._request(
                self._client_big, self._big_model, to_lang, query, '', 1.0, extra, refs)
        if trans is not None and self._gate_ok(query, trans):
            return trans
        if trans and self._has_hangul(trans):  # 과길이만 문제면 차선 허용
            return trans
        return None

    async def _glm(self, query: str) -> Optional[str]:
        """비상 폴백 — HYMT2_GL_FALLBACK=1 + CUSTOM_OPENAI_* 설정 시."""
        system = ('You are a manga dialogue translator. Translate each user text into natural '
                  'Korean matching the tone. Output only the translation.')
        try:
            async with self._sem:
                r = await self._client_glm.chat.completions.create(
                    model=self._glm_model,
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': query}],
                    temperature=0.3, max_tokens=512)
            trans = (r.choices[0].message.content or '').strip()
            return trans if self._gate_ok(query, trans) else None
        except Exception as e:
            self.logger.warning(f'GLM 폴백 실패: {e}')
            return None

    # ── 메인 ────────────────────────────────────────────────────

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        self._stat = {'tm': 0, 'vec_reuse': 0, 'vec_ref': 0, 'small': 0, 'big': 0, 'glm': 0,
                      'empty': 0, 'small_fail': 0, 'small_retry_ok': 0}
        results: List[Optional[str]] = [None] * len(queries)
        tier_of: Dict[int, str] = {}

        # Tier 0 — TM 정확일치 (LLM 호출 0)
        pending = []
        for i, q in enumerate(queries):
            hit = self._tm_lookup(q)
            if hit is not None:
                results[i] = hit
                tier_of[i] = 'tm'
                self._stat['tm'] += 1
            else:
                pending.append(i)

        # Tier 0.5 — 벡터 근접검색: 고유사도·고품질행 재사용, 그 아래는 참조 주입
        vec_refs: Dict[int, List[Tuple[str, str]]] = {}
        if pending and self._vec is not None and self._client_emb is not None:
            try:
                keys = [self._norm(queries[i]) for i in pending]
                for i, (sim, row) in zip(pending, await self._vec_search(keys)):
                    if sim < self._vec_ref_sim or not self._vec['trans'][row]:
                        continue
                    if sim >= self._vec_reuse_sim and self._vec['shares'][row] >= 0.75:
                        results[i] = self._vec['trans'][row]
                        tier_of[i] = 'vec'
                        self._stat['vec_reuse'] += 1
                    else:
                        vec_refs[i] = [(self._vec['keys'][row], self._vec['trans'][row])]
                        self._stat['vec_ref'] += 1
                pending = [i for i in pending if results[i] is None]
            except Exception as e:
                self.logger.warning(f'벡터 티어 건너뜀 (API?): {e}')

        # Tier 1 — 1.8B 전수 병렬 (소형 선처리: 대형은 이 웨이브가 끝난 뒤)
        if pending:
            small_results = await asyncio.gather(
                *(self._small(to_lang, queries[i], vec_refs.get(i)) for i in pending))
            for i, trans in zip(pending, small_results):
                if trans is not None:
                    results[i] = trans
                    tier_of[i] = 'small'
                    self._stat['small'] += 1
            pending = [i for i in pending if results[i] is None]

        # Tier 2 — 30B 실패분 일괄
        if pending and self._escalate_big:
            big_results = await asyncio.gather(
                *(self._big(to_lang, queries[i], vec_refs.get(i)) for i in pending))
            for i, trans in zip(pending, big_results):
                if trans is not None:
                    results[i] = trans
                    tier_of[i] = 'big'
                    self._stat['big'] += 1
            pending = [i for i in pending if results[i] is None]

        # 비상 폴백 — GLM 1회
        if pending and self._client_glm is not None:
            glm_results = await asyncio.gather(*(self._glm(queries[i]) for i in pending))
            for i, trans in zip(pending, glm_results):
                if trans is not None:
                    results[i] = trans
                    tier_of[i] = 'glm'
                    self._stat['glm'] += 1
            pending = [i for i in pending if results[i] is None]

        # 잔여 — 빈 칸 (부호뿉 원문은 원문 유지)
        for i in pending:
            results[i] = queries[i] if not self._needs_translation(queries[i]) else ''
            tier_of[i] = 'pass' if results[i] else 'empty'
            self._stat['empty'] += 1

        self._trace = [
            ('ref_' + tier_of[i] if i in vec_refs and tier_of[i] in ('small', 'big') else
             tier_of.get(i, 'empty'))
            for i in range(len(queries))
        ]

        n = len(queries)
        resolved_small = self._stat['tm'] + self._stat['vec_reuse'] + self._stat['small']
        self.logger.info(
            f'HyMT2 요약: 총 {n} — TM {self._stat["tm"]}, 벡터재사용 {self._stat["vec_reuse"]}, '
            f'1.8B {self._stat["small"]} (참조주입 {self._stat["vec_ref"]}, '
            f'재시도 회수 {self._stat["small_retry_ok"]}, 실패 {self._stat["small_fail"]}), '
            f'30B {self._stat["big"]}, GLM {self._stat["glm"]}, 빈칸 {self._stat["empty"]} '
            f'| ≤1.8B 해결률 {resolved_small * 100 // max(n, 1)}%')
        return [r or '' for r in results]

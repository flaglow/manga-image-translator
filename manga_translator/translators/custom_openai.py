import re

from ..config import TranslatorConfig
from .config_gpt import ConfigGPT  # Import the `gpt_config` parsing parent class

try:
    import openai
except ImportError:
    openai = None
import asyncio
import time
from typing import List
from .common import CommonTranslator, VALID_LANGUAGES
from .keys import (CUSTOM_OPENAI_API_KEY, CUSTOM_OPENAI_API_BASE, CUSTOM_OPENAI_MODEL, CUSTOM_OPENAI_MODEL_CONF,
                   CUSTOM_OPENAI_REASONING_EFFORT, CUSTOM_OPENAI_THINKING, CUSTOM_OPENAI_MAX_TOKENS,
                   CUSTOM_OPENAI_TEMPERATURE, CUSTOM_OPENAI_TIMEOUT, CUSTOM_OPENAI_MAX_CONCURRENCY)


class AdaptiveConcurrency:
    """429 백프레셔 컨트롤러 — LLM 동시 요청 수 조절.
    z.ai 429는 쿼터/동시성 기반이므로 effort(low 고정) 대신 동시성을 제어한다.
    429 시 동시 요청 -1 (최소 1), 20분 무429 시 +1 (설정 상한까지 복구)."""
    def __init__(self, initial: int):
        self.ceiling = max(1, initial)
        self.limit = self.ceiling
        self.in_flight = 0
        self.last_429 = 0.0
        self.cooldown = 20 * 60  # 회복 확인 주기
        self._cond = asyncio.Condition()

    async def acquire(self):
        """동시 요청 슬롯 확보. 복구 조건 충족 시 상한 회복 안내 반환."""
        async with self._cond:
            recovered = None
            if self.limit < self.ceiling and (time.time() - self.last_429) > self.cooldown:
                self.limit += 1
                recovered = f'적응형 동시성 복구 → {self.limit} (20분 무429)'
            await self._cond.wait_for(lambda: self.in_flight < self.limit)
            self.in_flight += 1
            self._cond.notify()
            return recovered

    async def release(self):
        async with self._cond:
            self.in_flight -= 1
            self._cond.notify()

    def on_429(self):
        self.last_429 = time.time()
        if self.limit > 1:
            self.limit -= 1
            return f'적응형 동시성 하향 → {self.limit} (429 백프레셔)'
        return None


_ADAPTIVE_CONC = AdaptiveConcurrency(CUSTOM_OPENAI_MAX_CONCURRENCY)


class CustomOpenAiTranslator(ConfigGPT, CommonTranslator):
    # 번역 거부 응답 감지용 스템 (전체 응답 거부 / 세그먼트 단위 거부).
    # 대사로도 쓰일 수 있는 표현("거부합니다", "미안하지만" 등)은 오탐 방지를 위해
    # 번역/요청/정책 같은 메타 단어를 동반한 형태만 등록한다.
    _REFUSAL_STEMS = (
        # 한국어 — "이 요청은 번역해 드릴 수 없습니다. 해당 텍스트는 ~ 이런 종류의 컨텐츠를 생성하거나 ~" 등
        '번역 요청은 진행할 수 없', '번역할 수 없', '번역해 드릴 수 없', '번역해드릴 수 없',
        '이 요청은 진행할 수 없', '이 요청은 번역', '해당 텍스트는', '해당 콘텐츠는', '해당 컨텐츠는',
        '이런 종류의 컨텐츠', '이런 종류의 콘텐츠', '이런 유형의 콘텐츠', '이런 유형의 컨텐츠',
        '컨텐츠를 생성', '콘텐츠를 생성', '유해한 내용', '부적절한 내용',
        '부적절한 콘텐츠', '부적절한 컨텐츠', '성인 콘텐츠', '성인 컨텐츠',
        '정책을 위반', '가이드라인을 위반', '정책상 이유', '이용 정책', '콘텐츠 정책', '컨텐츠 정책',
        # 영어
        "i'm sorry, but i can't", "i can't assist", 'i cannot assist',
        "i can't translate", 'i cannot translate',
        "i'm unable to", 'i am unable to', "i'm not able to", 'i am not able to',
        "i can't fulfill", 'i cannot fulfill', "i can't help with", 'i cannot help with',
        "i can't provide", 'i cannot provide', "i'm afraid i can't",
        'i must decline', 'against my guidelines', 'against our guidelines',
        'content policy', 'community guidelines', 'would violate', 'content boundaries',
        'inappropriate content', 'harmful content', 'as an ai language',
        # 일본어 — '申し訳ありませんが' 단독은 대사 오탐 위험으로 번역 동반 형태만
        '翻訳できません', 'お手伝いできません', '申し訳ありませんが、翻訳',
        '翻訳いたしかねます', '翻訳できかねます', '翻訳を拒否', '翻訳は控え', '不適切な内容', '不適切な表現',
        # 중국어
        '无法翻译', '无法协助', '我不能翻译', '无法提供翻译', '该文本包含', '不适当的内容', '违反相关政策',
    )
    # 세그먼트 품질(거부/과길이/누락) 문제 시 동일 프롬프트 재시도 상한
    _SEGMENT_RETRY_MAX = 2
    # 하드캡: 번역문 문자 수가 원문 문자 수의 이 배수 이상이면 재시도
    _LENGTH_HARD_CAP = 2

    # 품질 재시도 시 프롬프트 앞에 붙이는 수정 지시
    _HINT_REFUSAL = (
        '이전 응답에 번역 거부 문구("번역할 수 없습니다" 등)가 포함되어 있습니다. '
        '만화 대사 번역은 허용된 작업입니다. 안내문·사과·설명 없이 각 세그먼트의 번역 결과만 <|N|> 형식으로 출력하세요.'
    )
    _HINT_LENGTH = (
        '이전 번역이 원문보다 지나치게 깁니다. '
        '각 세그먼트를 원문 글자 수의 2배를 넘지 않도록, 말투는 유지하되 간결하게 번역하세요.'
    )

    @classmethod
    def _is_refusal(cls, response) -> bool:
        """GLM 등이 내놓는 '번역할 수 없습니다'류 응답을 감지.
        정상 응답에는 <|N|> 마커가 반드시 있으므로, 마커가 있으면 스트림 히트만 판정."""
        if not response:
            return False
        r = response.lower()
        return any(s in r for s in cls._REFUSAL_STEMS)

    @staticmethod
    def _char_len(s: str) -> int:
        """길이 비교용 문자 수 — 공백/줄바꿈은 세지 않는다."""
        return len(re.sub(r'\s+', '', s))

    @classmethod
    def _is_overlength(cls, query: str, trans: str) -> bool:
        """하드캡: 번역문 문자 수가 원문의 _LENGTH_HARD_CAP배 이상이면 True.
        원문이 1자('っ', '…' 등)인 경우 오탐이 잦아 제외한다."""
        q = cls._char_len(query)
        if q < 2:
            return False
        return cls._char_len(trans) >= cls._LENGTH_HARD_CAP * q

    _INVALID_REPEAT_COUNT = 2  # 如果检测到"无效"翻译，最多重复 2 次
    _MAX_REQUESTS_PER_MINUTE = 120  # 每分钟最大请求次数 (동시 배치 모드용으로 상향, 기존 40)
    _TIMEOUT = 40  # 在重试之前等待服务器响应的时间（秒）
    _RETRY_ATTEMPTS = 3  # 在放弃之前重试错误请求的次数
    _TIMEOUT_RETRY_ATTEMPTS = 3  # 在放弃之前重试超时请求的次数
    _RATELIMIT_RETRY_ATTEMPTS = 3  # 在放弃之前重试速率限制请求的次数

    # 最大令牌数量，用于控制处理的文本长度
    _MAX_TOKENS = 4096

    # 是否返回原始提示，用于控制输出内容
    _RETURN_PROMPT = False

    # 是否包含模板，用于决定是否使用预设的提示模板
    _INCLUDE_TEMPLATE = False

    def __init__(self, model=None, api_base=None, api_key=None, check_openai_key=False):
        # If the user has specified a nested key to use for the model, append the key
        #   Otherwise: Use the `ollama` defaults.
        _CONFIG_KEY='ollama'
        if CUSTOM_OPENAI_MODEL_CONF:
            _CONFIG_KEY+=f".{CUSTOM_OPENAI_MODEL_CONF}"

        ConfigGPT.__init__(self, config_key=_CONFIG_KEY)
        self.model = model
        CommonTranslator.__init__(self)
        self.client = openai.AsyncOpenAI(api_key=api_key or CUSTOM_OPENAI_API_KEY or "ollama") # required, but unused for ollama
        self.client.base_url = api_base or CUSTOM_OPENAI_API_BASE
        self.token_count = 0
        self.token_count_last = 0
        # Reasoning models (e.g. z.ai GLM with thinking enabled) can take far
        # longer than the default 40s to answer, so allow an env override.
        if CUSTOM_OPENAI_TIMEOUT:
            self._TIMEOUT = int(CUSTOM_OPENAI_TIMEOUT)

    def parse_args(self, args: TranslatorConfig):
        self.config = args.chatgpt_config


    def extract_capture_groups(self, text, regex=r"(.*)"):
        """
        Extracts all capture groups from matches and concatenates them into a single string.
        
        :param text: The multi-line text to search.
        :param regex: The regex pattern with capture groups.
        :return: A concatenated string of all matched groups.
        """
        pattern = re.compile(regex, re.DOTALL)  # DOTALL to match across multiple lines
        matches = pattern.findall(text)  # Find all matches
        
        # Ensure matches are concatonated (handles multiple groups per match)
        extracted_text = "\n".join(
            "\n".join(m) if isinstance(m, tuple) else m for m in matches
        )
        
        return extracted_text.strip() if extracted_text else None

    def _assemble_prompts(self, from_lang: str, to_lang: str, queries: List[str]):
        prompt = ''

        if self._INCLUDE_TEMPLATE:
            prompt += self.prompt_template.format(to_lang=to_lang)

        if self._RETURN_PROMPT:
            prompt += '\nOriginal:'

        i_offset = 0
        prompt_queries = []
        for i, query in enumerate(queries):
            prompt += f'\n<|{i + 1 - i_offset}|>{query}'
            prompt_queries.append(query)

            # If prompt is growing too large and there's still a lot of text left
            # split off the rest of the queries into new prompts.
            # 1 token = ~4 characters according to https://platform.openai.com/tokenizer
            # TODO: potentially add summarizations from special requests as context information
            if self._MAX_TOKENS * 2 and len(''.join(queries[i + 1:])) > self._MAX_TOKENS:
                if self._RETURN_PROMPT:
                    prompt += '\n<|1|>'
                yield prompt.lstrip(), i + 1 - i_offset, prompt_queries
                prompt = self.prompt_template.format(to_lang=to_lang)
                prompt_queries = []
                # Restart counting at 1
                i_offset = i + 1

        if self._RETURN_PROMPT:
            prompt += '\n<|1|>'

        yield prompt.lstrip(), len(queries) - i_offset, prompt_queries

    def _format_prompt_log(self, to_lang: str, prompt: str) -> str:
        if to_lang in self.chat_sample:
            return '\n'.join([
                'System:',
                self.chat_system_template.format(to_lang=to_lang),
                'User:',
                self.chat_sample[to_lang][0],
                'Assistant:',
                self.chat_sample[to_lang][1],
                'User:',
                prompt,
            ])
        else:
            return '\n'.join([
                'System:',
                self.chat_system_template.format(to_lang=to_lang),
                'User:',
                prompt,
            ])

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        self.logger.debug(f'Temperature: {self.temperature}, TopP: {self.top_p}')

        for prompt, query_size, prompt_queries in self._assemble_prompts(from_lang, to_lang, queries):
            self.logger.debug('-- GPT Prompt --\n' + self._format_prompt_log(to_lang, prompt))

            # 세그먼트별 확정값(품질 검사 통과) / 과길이였던 최단 후보(재시도 소진 시 차선)
            best = [None] * query_size
            fallback = [None] * query_size
            hint = ''
            quality_attempt = 0

            while True:
                response = await self._request_with_retry(to_lang, prompt, hint)

                # 응답 전체가 거부 메시지(<|N|> 마커 없음)인 경우
                if self._is_refusal(response) and not re.search(r'<\|?\s*\d+\s*\|?>', response):
                    if quality_attempt < self._SEGMENT_RETRY_MAX:
                        quality_attempt += 1
                        self.logger.warning(f'번역 거부 응답 감지 — 재시도 {quality_attempt}/{self._SEGMENT_RETRY_MAX}')
                        hint = self._HINT_REFUSAL
                        await asyncio.sleep(1)
                        continue
                    self.logger.warning('번역 거부 지속 — 해당 세그먼트들을 빈 칸으로 대체합니다')
                    break

                segments = self._parse_segments(response, query_size)

                # 세그먼트별 검증: 거부 문구 / 누락 / 원문 2배 이상 과길이.
                # 이미 확정된 세그먼트는 그대로 유지한다(기존 번역 보존).
                bad_refusal = bad_empty = bad_overlength = 0
                for i, (query, trans) in enumerate(zip(prompt_queries, segments)):
                    if best[i] is not None:
                        continue
                    if self._is_refusal(trans):
                        bad_refusal += 1
                        continue
                    if not trans:
                        bad_empty += 1
                        continue
                    if self._is_overlength(query, trans):
                        bad_overlength += 1
                        if fallback[i] is None or self._char_len(trans) < self._char_len(fallback[i]):
                            fallback[i] = trans
                        continue
                    best[i] = trans

                if all(v is not None for v in best):
                    break
                if quality_attempt >= self._SEGMENT_RETRY_MAX:
                    self.logger.warning(
                        f'품질 재시도 소진 (거부 {bad_refusal}, 누락 {bad_empty}, 과길이 {bad_overlength}) — '
                        '거부/누락 세그먼트는 빈 칸, 과길이 세그먼트는 최단 후보로 대체합니다')
                    break

                quality_attempt += 1
                hint = '\n'.join(h for h, bad in (
                    (self._HINT_REFUSAL, bad_refusal),
                    (self._HINT_LENGTH, bad_overlength),
                ) if bad)
                self.logger.warning(
                    f'세그먼트 품질 문제 (거부 {bad_refusal}, 누락 {bad_empty}, 과길이 {bad_overlength}) — '
                    f'재시도 {quality_attempt}/{self._SEGMENT_RETRY_MAX}')
                await asyncio.sleep(1)

            # 미확정 세그먼트: 과길이 최단 후보 → 없으면 빈 칸(거부 문구 렌더링 방지)
            translations.extend(best[i] if best[i] is not None else (fallback[i] or '')
                                for i in range(query_size))

        # 최종 안전 스윕 — 어떤 경로로든 거부 문구가 남으면 이미지에 들어가지 않도록 제거
        for i, t in enumerate(translations):
            if t and self._is_refusal(t):
                self.logger.warning(f'최종 스윕: 거부 문구 제거 (segment {i}: {t[:40]}...)')
                translations[i] = ''

        self.logger.debug(translations)
        if self.token_count_last:
            self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')

        return translations

    async def _request_with_retry(self, to_lang: str, prompt: str, hint: str = '') -> str:
        """타임아웃/레이트리밋/서버오류 재시도를 포함해 1회 번역 요청을 수행한다."""
        ratelimit_attempt = 0
        server_error_attempt = 0
        timeout_attempt = 0
        while True:
            request_task = asyncio.create_task(self._request_translation(to_lang, prompt, hint))
            started = time.time()
            while not request_task.done():
                await asyncio.sleep(0.1)
                if time.time() - started > self._TIMEOUT + (timeout_attempt * self._TIMEOUT / 2):
                    # Server takes too long to respond
                    if timeout_attempt >= self._TIMEOUT_RETRY_ATTEMPTS:
                        raise Exception('ollama servers did not respond quickly enough.')
                    timeout_attempt += 1
                    self.logger.warning(f'Restarting request due to timeout. Attempt: {timeout_attempt}')
                    request_task.cancel()
                    request_task = asyncio.create_task(self._request_translation(to_lang, prompt, hint))
                    started = time.time()
            try:
                return await request_task
            except openai.RateLimitError:  # Server returned ratelimit response
                note = _ADAPTIVE_CONC.on_429()
                if note:
                    self.logger.warning(note)
                ratelimit_attempt += 1
                if ratelimit_attempt >= self._RATELIMIT_RETRY_ATTEMPTS:
                    raise
                self.logger.warning(
                    f'Restarting request due to ratelimiting by Ollama servers. Attempt: {ratelimit_attempt}')
                await asyncio.sleep(2)
            except openai.APIError:  # Server returned 500 error (probably server load)
                server_error_attempt += 1
                if server_error_attempt >= self._RETRY_ATTEMPTS:
                    self.logger.error(
                        'Ollama encountered a server error, possibly due to high server load. Use a different translator or try again later.')
                    raise
                self.logger.warning(f'Restarting request due to a server error. Attempt: {server_error_attempt}')
                await asyncio.sleep(1)

    def _parse_segments(self, response: str, query_size: int) -> List[str]:
        """응답 원문을 query_size 길이의 세그먼트 리스트로 정렬·정돈한다."""
        if not response:
            return [''] * query_size

        # Use regex to extract response
        response = self.extract_capture_groups(response, rf"{self.rgx_capture}")

        # Sometimes it will return line like "<|9>demo", and we need to fix it.
        def add_pipe(match):
            number = match.group(1)
            return f"<|{number}|>"
        response = re.sub(r"<\|?(\d+)\|?>", add_pipe, response)

        # Remove any text preceeding the first translation.
        new_translations = re.split(r'<\|\d+\|>', 'pre_1\n' + response)[1:]

        # When there is only one query LLMs likes to exclude the <|1|>
        if not new_translations:
            new_translations = [response]

        # Immediately clean leading and trailing whitespace from each translation text
        new_translations = [t.strip() for t in new_translations]

        # Maybe it can be removed, but it causes no errors
        if not new_translations[0].strip():
            new_translations = new_translations[1:]

        if len(new_translations) <= 1 and query_size > 1:
            # Try splitting by newlines instead
            new_translations = re.split(r'\n', response)

        if len(new_translations) > query_size:
            new_translations = new_translations[: query_size]
        elif len(new_translations) < query_size:
            new_translations = new_translations + [''] * (query_size - len(new_translations))

        # <|N|> 마커가 변형 형태(전각 파이프/공백 포함)로 세그먼트 안에 남아
        # 이미지에 그대로 렌더링되는 사고 방지 — 최종 단계에서 전부 제거
        def _strip_markers(t: str) -> str:
            cleaned = re.sub(r'[<＜]\s*[|｜]*\s*\d+\s*[|｜]*\s*[>＞]', ' ', t)
            return re.sub(r'\s{2,}', ' ', cleaned).strip()

        return [_strip_markers(t) for t in new_translations]

    async def _request_translation(self, to_lang: str, prompt: str, hint: str = '') -> str:
        messages = [{'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}]

        # Add chat samples if available
        lang_chat_samples = self.get_chat_sample(to_lang)
        if lang_chat_samples:
            messages.append({'role': 'user', 'content': lang_chat_samples[0]})
            messages.append({'role': 'assistant', 'content': lang_chat_samples[1]})

        # 품질 재시도 시 프롬프트 앞에 수정 지시(거부/과길이 힌트)를 붙인다
        if hint:
            self.logger.warning(f'Retry hint appended to prompt: {hint[:60]}...')
        messages.append({'role': 'user', 'content': (hint + '\n' + prompt) if hint else prompt})

        max_tokens = self._MAX_TOKENS // 2
        if CUSTOM_OPENAI_MAX_TOKENS:
            max_tokens = int(CUSTOM_OPENAI_MAX_TOKENS)
        temperature = self.temperature
        if CUSTOM_OPENAI_TEMPERATURE:
            temperature = float(CUSTOM_OPENAI_TEMPERATURE)

        # Pass provider-specific reasoning parameters (e.g. z.ai GLM
        # `thinking` / `reasoning_effort`) through extra_body so they reach the
        # request JSON regardless of the openai SDK version in use.
        extra_body = {}
        if CUSTOM_OPENAI_THINKING:
            extra_body['thinking'] = {'type': CUSTOM_OPENAI_THINKING}
        if CUSTOM_OPENAI_REASONING_EFFORT:
            extra_body['reasoning_effort'] = CUSTOM_OPENAI_REASONING_EFFORT

        # 적응형 동시성 게이트 — 실제 HTTP 요청을 슬롯 수만큼만 동시 진행
        note = await _ADAPTIVE_CONC.acquire()
        if note:
            self.logger.warning(note)
        try:
            response = await self.client.chat.completions.create(
                model=self.model or CUSTOM_OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=self.top_p,
                extra_body=extra_body,
            )
        finally:
            await _ADAPTIVE_CONC.release()

        self.logger.debug('\n-- GPT Response (raw) --')
        self.logger.debug(response.choices[0].message.content)
        self.logger.debug('------------------------\n')


        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        return response.choices[0].message.content

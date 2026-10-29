import asyncio
import os
import uuid
from typing import List

from .common import CommonTranslator


class HdTransTranslator(CommonTranslator):
    """hdTrans — 룰 기반 오프라인 일한 번역기 (sokcuri/hdTrans, MPL-2.0).

    CLI를 페이지 단위로 호출한다(사전 로드 0.3초, 세그먼트당 ~2ms — 상시 프로세스·GPU 불필요).
    --each 로 세그먼트(줄)를 독립 번역하고, --speech soft 로 대사를 반말(해체)로 쓴다.
    사용자 사전(configs/hdtrans-userdict.txt)은 프로덕션 GLM 로그 코퍼스에서
    마이닝해 정렬해 둔 것 — 갱신 절차는 그 파일 헤더 참조.

    GLM 경로(gpt-kor.toml)는 그대로 두고, hd-kor.toml의 translator 설정으로 선택한다.
    배포: vendor/hdtrans/hdTrans-0.7.4/ 를 컨테이너에 마운트(compose · FINISH_MOUNTS).
    재정의: HDTRANS_PATH(바이너리), HDTRANS_USERDICT, HDTRANS_SPEECH, HDTRANS_CONCURRENCY
    """
    _LANGUAGE_CODE_MAP = {'JPN': 'Japanese', 'KOR': 'Korean'}

    _TIMEOUT = 120  # 페이지 단위 배치(수십 줄)면 수 초 이내

    _CANDIDATE_PATHS = (
        os.environ.get('HDTRANS_PATH', ''),
        '/app/vendor/hdtrans/hdTrans-0.7.4/hdtrans',   # 컨테이너 마운트 기본
        'vendor/hdtrans/hdTrans-0.7.4/hdtrans',        # 저장소 루트 실행 시
    )

    def __init__(self):
        super().__init__()
        self._bin = next((p for p in self._CANDIDATE_PATHS
                          if p and os.path.isfile(p) and os.access(p, os.X_OK)), None)
        if self._bin is None:
            raise RuntimeError(
                'hdtrans 바이너리를 찾을 수 없음 — vendor/hdtrans/ 배치 또는 HDTRANS_PATH 설정')
        self._speech = os.environ.get('HDTRANS_SPEECH', 'soft')
        userdict = os.environ.get('HDTRANS_USERDICT', 'configs/hdtrans-userdict.txt')
        self._userdict = userdict if os.path.isfile(userdict) else None
        self._sem = asyncio.Semaphore(int(os.environ.get('HDTRANS_CONCURRENCY', '4')))

    def parse_args(self, args):
        pass  # toml 확장 필드 없음 — env로만 재정의

    async def _run_batch(self, lines: List[str]) -> List[str]:
        """줄 목록을 hdtrans --each 로 번역해 같은 순서로 돌려준다."""
        token = uuid.uuid4().hex[:8]
        in_path = f'/tmp/hdtrans-{token}.in'
        out_path = f'/tmp/hdtrans-{token}.out'
        cmd = [self._bin, '--each', '--speech', self._speech]
        if self._userdict:
            cmd += ['--userdict', self._userdict]
        cmd += ['--in', in_path, '--out', out_path]
        # 파이프라인 이미지의 libstdc++가 낡아(GLIBCXX_3.4.31 부재) 바이너리 옆 lib/ 를 우선 로드
        env = None
        lib_dir = os.path.join(os.path.dirname(self._bin), 'lib')
        if os.path.isdir(lib_dir):
            env = dict(os.environ)
            env['LD_LIBRARY_PATH'] = lib_dir + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        try:
            with open(in_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._TIMEOUT)
            if proc.returncode != 0:
                raise RuntimeError(f'hdtrans 실패(exit {proc.returncode}): {stderr.decode(errors="replace")[:500]}')
            with open(out_path, encoding='utf-8') as f:
                outs = [l.rstrip('\n') for l in f]
        finally:
            for p in (in_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if len(outs) != len(lines):
            raise RuntimeError(f'hdtrans 출력 줄 수 불일치: {len(outs)} != {len(lines)}')
        return outs

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        """CommonTranslator.translate 템플릿에서 호출 — 빈 세그먼트는 이미 걸러져 온다."""
        lines = [q.replace('\n', ' ').replace('\t', ' ').strip() for q in queries]
        async with self._sem:
            try:
                return await self._run_batch(lines)
            except Exception as e:
                self.logger.warn(f'hdtrans 배치 재시도: {e}')
                return await self._run_batch(lines)  # 재시도 실패 시 페이지 실패로 전파

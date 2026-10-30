# 파이프라인 이미지에는 requirements-dev(pytest)가 없어, 즉석 설치 후 테스트를 돌린다.
# 사용: docker run --rm --entrypoint python -v "$PWD:/app" -w /app \
#        zyddnys/manga-image-translator:main test/run_hymt2_in_docker.py
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'pytest==7.2.2', 'pytest-asyncio==0.21.1'], check=True)
raise SystemExit(subprocess.run(
    [sys.executable, '-m', 'pytest', 'test/test_hymt2.py', '-q', '--no-header']).returncode)

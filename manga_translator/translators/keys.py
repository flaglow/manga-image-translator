import os
from dotenv import load_dotenv
load_dotenv()

# baidu
BAIDU_APP_ID = os.getenv('BAIDU_APP_ID', '') #你的appid
BAIDU_SECRET_KEY = os.getenv('BAIDU_SECRET_KEY', '') #你的密钥
# youdao
YOUDAO_APP_KEY = os.getenv('YOUDAO_APP_KEY', '') # 应用ID
YOUDAO_SECRET_KEY = os.getenv('YOUDAO_SECRET_KEY', '') # 应用秘钥
# deepl
DEEPL_AUTH_KEY = os.getenv('DEEPL_AUTH_KEY', '') #YOUR_AUTH_KEY
# openai
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-5.4-mini')

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')

OPENAI_HTTP_PROXY = os.getenv('OPENAI_HTTP_PROXY') # TODO: Replace with --proxy
OPENAI_GLOSSARY_PATH = os.getenv('OPENAI_GLOSSARY_PATH', './dict/mit_glossary.txt') # OpenAI术语表路径
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1') #使用api-for-open-llm例子 http://127.0.0.1:8000/v1

# sakura
SAKURA_API_BASE = os.getenv('SAKURA_API_BASE', 'http://127.0.0.1:8080/v1') #SAKURA API地址
SAKURA_VERSION = os.getenv('SAKURA_VERSION', '0.9') #SAKURA API版本，可选值：0.9、0.10，选择0.10则会加载术语表。
SAKURA_DICT_PATH = os.getenv('SAKURA_DICT_PATH', './dict/sakura_dict.txt') #SAKURA 术语表路径


CAIYUN_TOKEN = os.getenv('CAIYUN_TOKEN', '') # 彩云小译API访问令牌

# Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash-002')

# deepseek
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_BASE  = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com')
DEEPSEEK_MODEL  = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat') # Or: "deepseek-reasoner"

# Together AI
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY', '')
TOGETHER_VL_MODEL = os.getenv('TOGETHER_VL_MODEL', 'Qwen/Qwen2.5-VL-72B-Instruct')

# ollama, with OpenAI API compatibility
CUSTOM_OPENAI_API_KEY = os.getenv('CUSTOM_OPENAI_API_KEY', 'ollama') # Unsed for ollama, but maybe useful for other LLM tools.
CUSTOM_OPENAI_API_BASE = os.getenv('CUSTOM_OPENAI_API_BASE', 'http://localhost:11434/v1') # Use OLLAMA_HOST env to change binding IP and Port.
CUSTOM_OPENAI_MODEL = os.getenv('CUSTOM_OPENAI_MODEL', '') # e.g "qwen2.5:7b". Make sure to pull and run it before use.
CUSTOM_OPENAI_MODEL_CONF = os.getenv('CUSTOM_OPENAI_MODEL_CONF', '') # e.g "qwen2".
# Optional request parameters for reasoning models / OpenAI-compatible providers (e.g. z.ai GLM)
CUSTOM_OPENAI_REASONING_EFFORT = os.getenv('CUSTOM_OPENAI_REASONING_EFFORT', '') # e.g "max" | "high" | "low"
CUSTOM_OPENAI_THINKING = os.getenv('CUSTOM_OPENAI_THINKING', '') # e.g "enabled" | "disabled" (z.ai thinking.type)
CUSTOM_OPENAI_MAX_TOKENS = os.getenv('CUSTOM_OPENAI_MAX_TOKENS', '') # override max_tokens (default: _MAX_TOKENS // 2)
CUSTOM_OPENAI_TEMPERATURE = os.getenv('CUSTOM_OPENAI_TEMPERATURE', '') # override temperature (default: chatgpt config / 0.5)
CUSTOM_OPENAI_TIMEOUT = os.getenv('CUSTOM_OPENAI_TIMEOUT', '') # per-request timeout seconds (default: 40)
try:
    CUSTOM_OPENAI_MAX_CONCURRENCY = int(os.getenv('CUSTOM_OPENAI_MAX_CONCURRENCY') or 5) # LLM 동시 요청 상한 (계정 한계 5, 429 백프레셔 시작값)
except ValueError:
    CUSTOM_OPENAI_MAX_CONCURRENCY = 5

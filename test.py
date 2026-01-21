import json
import os
import sys
from pathlib import Path

from openai import OpenAI


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[len("export ") :].strip()
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def call_llm(content: str) -> dict:
    _load_env_file(Path(__file__).with_name(".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    if not api_key or not base_url:
        raise SystemExit("Missing OPENAI_API_KEY or OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)

    text = content if content and content.strip() else " "
    messages = [{"role": "user", "content": text}]
    response = client.chat.completions.create(model=model, messages=messages)
    try:
        return response.model_dump()
    except Exception:
        return {"raw": str(response)}


def main() -> None:
    content = " ".join(sys.argv[1:]).strip()
    result = call_llm(content)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

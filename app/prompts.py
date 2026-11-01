from app import config


def load_prompt(name: str) -> str:
    path = config.PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def list_prompts(prefix: str) -> list[str]:
    return sorted(p.stem for p in config.PROMPTS_DIR.glob(f"{prefix}*.txt"))

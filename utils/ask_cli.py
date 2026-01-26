def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [default: {default}]" if default is not None else ""
    value = input(f"---\n{prompt}{suffix}: ").strip()
    return value if value else (default or "")


def ask_int(prompt: str, default: int = 5) -> int:
    while True:
        s = ask(prompt, str(default))
        try:
            return int(s)
        except ValueError:
            print("Please enter a valid integer.")


def ask_list(prompt: str, default_csv: str) -> list[str]:
    s = ask(prompt, default_csv)
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items

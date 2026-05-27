"""
Monte-Carlo-Latenz-Benchmark für die Claude API.

Sampelt zufällige Kombinationen aus (Modell, Prompt-Kategorie, Prompt-Größe, max_tokens),
führt API-Calls aus und berichtet Latenz-Verteilungen (Mean, Median, P95, P99).

Voraussetzungen:
    pip install -r requirements.txt
    .env-Datei anlegen mit: ANTHROPIC_API_KEY=sk-ant-...

Ausführen:
    python claude_api_mc_benchmark.py

Achtung: Mit N=60 Iterationen und gemischten Prompt-Größen können je nach
Sampling ein paar Cent bis ein paar Dollar API-Kosten anfallen.
"""

import csv
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()  # liest ANTHROPIC_API_KEY aus .env oder Umgebung

# ----- Sampling-Räume --------------------------------------------------------

MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
]

# Prompt-Größen in Wörtern (1 Wort ≈ 1.3 Tokens grob)
PROMPT_SIZES = {
    "tiny":   10,
    "small":  100,
    "medium": 500,
    "large":  2000,
    "xlarge": 8000,
}

MAX_TOKENS_OPTIONS = [50, 200, 500, 1000]

# ----- Prompt-Generatoren pro Kategorie --------------------------------------

_PROSE_WORDS = (
    "der die das ein eine ist hat wird war wurde haben werden sein können müssen "
    "sollen wollen Zeit Mensch Leben Welt Jahr Tag Haus Land Mann Frau Kind "
    "Arbeit Stadt Hand Schule Wasser Geld Problem Frage Möglichkeit Entwicklung "
    "Gesellschaft Situation Bereich System Ergebnis Bedeutung Aufgabe Prozess"
).split()

_CODE_NAMES = (
    "data result value item count index key node tree graph config "
    "settings output input buffer cache handler parser builder manager"
).split()
_CODE_TYPES = "int str list dict bool float None bytes".split()


def _make_prose(word_count: int, rng: random.Random) -> str:
    words = (_PROSE_WORDS * (word_count // len(_PROSE_WORDS) + 1))[:word_count]
    rng.shuffle(words)
    return " ".join(words) + "\n\nFasse den obigen Text in einem Satz auf Deutsch zusammen."


def _make_code(word_count: int, rng: random.Random) -> str:
    lines = []
    words_used = 0
    while words_used < word_count:
        name = rng.choice(_CODE_NAMES)
        other = rng.choice(_CODE_NAMES)
        t = rng.choice(_CODE_TYPES)
        line = rng.choice([
            f"def {name}({other}: {t}) -> {rng.choice(_CODE_TYPES)}:",
            f"    {name} = {other}.{rng.choice(_CODE_NAMES)}()",
            f"    return {name}",
            f"{name} = [{other} for {other} in range({rng.randint(1, 100)})]",
            f"if {name} is not None:",
            f"    {name}.append({other})",
            f"{name}: {t} = {rng.choice([str(rng.randint(0, 999)), repr(''), 'None', '[]', '{}'])}",
        ])
        lines.append(line)
        words_used += len(line.split())
    code = "\n".join(lines)
    return f"```python\n{code}\n```\n\nErkläre kurz auf Deutsch, was dieser Code macht."


def _make_json(word_count: int, rng: random.Random) -> str:
    _keys = (
        "id name type value status config data result error message "
        "count total active enabled version created updated"
    ).split()

    def _val():
        c = rng.randint(0, 3)
        if c == 0:
            return rng.randint(0, 9999)
        if c == 1:
            return rng.choice(["active", "inactive", "pending", "error"])
        if c == 2:
            return bool(rng.randint(0, 1))
        return f"item_{rng.randint(100, 999)}"

    def _obj(depth: int = 0) -> dict:
        o = {}
        for _ in range(rng.randint(3, 6)):
            k = rng.choice(_keys)
            o[k] = _obj(depth + 1) if depth < 2 and rng.random() > 0.65 else _val()
        return o

    blocks, total = [], 0
    while total < word_count:
        s = json.dumps(_obj(), indent=2)
        blocks.append(s)
        total += len(s.split())
    return "\n".join(blocks) + "\n\nListe alle vorhandenen Schlüssel (keys) auf."


def _make_instruction(word_count: int, rng: random.Random) -> str:
    _topics = [
        "Python-Skript", "REST-API", "Datenbank-Schema", "CI/CD-Pipeline",
        "Docker-Container", "Machine-Learning-Modell", "Webserver", "CLI-Tool",
        "Monitoring-System", "Authentifizierungs-Service",
    ]
    _actions = [
        "implementieren", "optimieren", "debuggen",
        "dokumentieren", "testen", "deployen",
    ]
    _details = [
        "Berücksichtige dabei Performance und Sicherheit.",
        "Achte auf Fehlerbehandlung und Logging.",
        "Schreibe Tests für alle Kernfunktionen.",
        "Halte die Abhängigkeiten minimal.",
        "Dokumentiere alle öffentlichen Schnittstellen.",
    ]
    steps, words_used, n = [], 0, 1
    while words_used < word_count:
        step = f"{n}. {rng.choice(_topics)} {rng.choice(_actions)}: {rng.choice(_details)}"
        steps.append(step)
        words_used += len(step.split())
        n += 1
    return "\n".join(steps) + "\n\nWelcher Schritt ist am komplexesten? Antworte in einem Satz."


PROMPT_CATEGORIES = {
    "prose":       _make_prose,
    "code":        _make_code,
    "json":        _make_json,
    "instruction": _make_instruction,
}

# ----- Daten-Container -------------------------------------------------------

@dataclass
class Sample:
    iteration: int
    model: str
    category: str
    prompt_size: str
    word_count: int
    max_tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    ttft: float = float("nan")
    total: float = float("nan")
    tps: float = 0.0
    error: str = ""


# ----- Einzelner API-Call ----------------------------------------------------

def single_run(it: int, model: str, category: str, size_name: str,
               word_count: int, max_tokens: int, rng: random.Random) -> Sample:
    prompt = PROMPT_CATEGORIES[category](word_count, rng)
    sample = Sample(
        iteration=it, model=model, category=category,
        prompt_size=size_name, word_count=word_count, max_tokens=max_tokens,
    )

    start = time.perf_counter()
    first_at = None

    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for event in stream:
                if first_at is None and event.type == "content_block_delta":
                    first_at = time.perf_counter()
            final = stream.get_final_message()

        end = time.perf_counter()
        sample.total = end - start
        sample.ttft = (first_at - start) if first_at else float("nan")
        sample.input_tokens = final.usage.input_tokens
        sample.output_tokens = final.usage.output_tokens

        gen_time = end - (first_at or start)
        sample.tps = sample.output_tokens / gen_time if gen_time > 0 else 0.0

    except Exception as e:
        sample.error = type(e).__name__ + ": " + str(e)[:100]

    return sample


# ----- Monte-Carlo-Loop ------------------------------------------------------

def monte_carlo(n_iterations: int = 60, seed: int = 42,
                pause_range=(0.1, 0.4)) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []

    print(f"Monte-Carlo-Simulation: {n_iterations} Iterationen, seed={seed}\n")

    for i in range(n_iterations):
        model = rng.choice(MODELS)
        size_name, word_count = rng.choice(list(PROMPT_SIZES.items()))
        max_tok = rng.choice(MAX_TOKENS_OPTIONS)
        category = rng.choice(list(PROMPT_CATEGORIES.keys()))

        s = single_run(i, model, category, size_name, word_count, max_tok, rng)
        samples.append(s)

        status = "ERR" if s.error else "OK "
        print(f"  [{i + 1:3d}/{n_iterations}] {status} {s.model:18s} "
              f"cat={s.category:11s} size={s.prompt_size:6s} max_tok={s.max_tokens:4d} "
              f"in={s.input_tokens:5d} out={s.output_tokens:4d} "
              f"total={s.total:6.2f}s ttft={s.ttft:5.2f}s tps={s.tps:5.1f}")

        time.sleep(rng.uniform(*pause_range))

    return samples


# ----- Auswertung ------------------------------------------------------------

def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _line(label: str, values: list[float]) -> str:
    if not values:
        return f"  {label:22s}  n=0"
    n = len(values)
    mean = statistics.mean(values)
    med = statistics.median(values)
    p95 = percentile(values, 95)
    p99 = percentile(values, 99)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return (f"  {label:22s}  n={n:3d}  mean={mean:6.2f}  med={med:6.2f}  "
            f"p95={p95:6.2f}  p99={p99:6.2f}  sd={sd:5.2f}")


def report(samples: list[Sample]) -> None:
    ok = [s for s in samples if not s.error]
    errs = [s for s in samples if s.error]

    print(f"\n{'=' * 78}")
    print(f"  Auswertung: {len(ok)} erfolgreich / {len(errs)} Fehler")
    print('=' * 78)

    print("\nGesamt:")
    print(_line("total (s)", [s.total for s in ok]))
    print(_line("TTFT (s)", [s.ttft for s in ok if not (s.ttft != s.ttft)]))
    print(_line("tokens/sek (output)", [s.tps for s in ok]))

    print("\nPro Modell — TTFT (s):")
    for m in MODELS:
        print(_line(m, [s.ttft for s in ok if s.model == m]))

    print("\nPro Modell — Total (s):")
    for m in MODELS:
        print(_line(m, [s.total for s in ok if s.model == m]))

    print("\nPro Kategorie — TTFT (s):")
    for cat in PROMPT_CATEGORIES:
        print(_line(cat, [s.ttft for s in ok if s.category == cat]))

    print("\nPro Kategorie — Total (s):")
    for cat in PROMPT_CATEGORIES:
        print(_line(cat, [s.total for s in ok if s.category == cat]))

    print("\nPro Prompt-Größe — TTFT (s):")
    for size in PROMPT_SIZES:
        print(_line(size, [s.ttft for s in ok if s.prompt_size == size]))

    if errs:
        print("\nFehler-Beispiele:")
        for e in errs[:5]:
            print(f"  - {e.model} {e.category} {e.prompt_size}: {e.error}")


def save_csv(samples: list[Sample], path: str) -> None:
    if not samples:
        return
    fields = list(asdict(samples[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))
    print(f"\nRohdaten → {path}")


# ----- Plots -----------------------------------------------------------------

def plot_results(samples: list[Sample], path: str) -> None:
    ok = [s for s in samples if not s.error and s.ttft == s.ttft]
    if not ok:
        print("Keine erfolgreichen Samples zum Plotten.")
        return

    models_present = [m for m in MODELS if any(s.model == m for s in ok)]
    sizes_present = [sz for sz in PROMPT_SIZES if any(s.prompt_size == sz for s in ok)]
    cats_present = [c for c in PROMPT_CATEGORIES if any(s.category == c for s in ok)]
    color_map = {m: c for m, c in zip(MODELS, ["#4C72B0", "#DD8452", "#55A467"])}
    cat_colors = ["#9B59B6", "#E74C3C", "#1ABC9C", "#F39C12"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Claude API Latenz-Benchmark — N={len(ok)} Samples",
                 fontsize=14, fontweight="bold")

    # 1) Boxplot: TTFT pro Modell
    ax = axes[0, 0]
    data = [[s.ttft for s in ok if s.model == m] for m in models_present]
    bp = ax.boxplot(data, labels=models_present, patch_artist=True, showfliers=True)
    for patch, m in zip(bp["boxes"], models_present):
        patch.set_facecolor(color_map[m])
        patch.set_alpha(0.7)
    ax.set_title("TTFT pro Modell")
    ax.set_ylabel("Time To First Token (s)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)

    # 2) Boxplot: Total-Latenz pro Modell
    ax = axes[0, 1]
    data = [[s.total for s in ok if s.model == m] for m in models_present]
    bp = ax.boxplot(data, labels=models_present, patch_artist=True, showfliers=True)
    for patch, m in zip(bp["boxes"], models_present):
        patch.set_facecolor(color_map[m])
        patch.set_alpha(0.7)
    ax.set_title("Gesamt-Latenz pro Modell")
    ax.set_ylabel("Total Latency (s)")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.3)

    # 3) Boxplot: TTFT pro Prompt-Größe
    ax = axes[0, 2]
    data = [[s.ttft for s in ok if s.prompt_size == sz] for sz in sizes_present]
    ax.boxplot(data, labels=sizes_present, patch_artist=True,
               boxprops=dict(facecolor="#888", alpha=0.6), showfliers=True)
    ax.set_title("TTFT pro Prompt-Größe")
    ax.set_ylabel("TTFT (s)")
    ax.set_xlabel("Prompt-Größe")
    ax.grid(True, alpha=0.3)

    # 4) Boxplot: TTFT pro Kategorie
    ax = axes[1, 0]
    data = [[s.ttft for s in ok if s.category == c] for c in cats_present]
    bp = ax.boxplot(data, labels=cats_present, patch_artist=True, showfliers=True)
    for patch, color in zip(bp["boxes"], cat_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("TTFT pro Kategorie")
    ax.set_ylabel("TTFT (s)")
    ax.grid(True, alpha=0.3)

    # 5) Scatter: Output-Tokens vs Total-Latenz
    ax = axes[1, 1]
    for m in models_present:
        xs = [s.output_tokens for s in ok if s.model == m]
        ys = [s.total for s in ok if s.model == m]
        ax.scatter(xs, ys, label=m, color=color_map[m], alpha=0.7, s=40)
    ax.set_title("Output-Tokens → Total-Latenz")
    ax.set_xlabel("Output-Tokens")
    ax.set_ylabel("Total Latency (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6) Bar: Throughput pro Modell
    ax = axes[1, 2]
    means, meds, labels = [], [], []
    for m in models_present:
        vals = [s.tps for s in ok if s.model == m and s.tps > 0]
        if vals:
            means.append(statistics.mean(vals))
            meds.append(statistics.median(vals))
            labels.append(m)
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], means, width=0.4, label="Mean",
           color=[color_map[m] for m in labels], alpha=0.9)
    ax.bar([i + 0.2 for i in x], meds, width=0.4, label="Median",
           color=[color_map[m] for m in labels], alpha=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title("Output-Throughput pro Modell")
    ax.set_ylabel("Tokens / Sekunde")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plots → {path}")


# ----- Main ------------------------------------------------------------------

if __name__ == "__main__":
    samples = monte_carlo(n_iterations=60, seed=42)
    report(samples)
    save_csv(samples, "claude_api_mc_results.csv")
    plot_results(samples, "claude_api_mc_plots.png")

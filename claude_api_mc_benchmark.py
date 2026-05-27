"""
Monte-Carlo-Latenz-Benchmark für die Claude API.

Sampelt zufällige Kombinationen aus (Modell, Prompt-Größe, max_tokens),
führt API-Calls aus und berichtet Latenz-Verteilungen (Mean, Median, P95, P99).

Voraussetzungen:
    pip install anthropic python-dotenv
    .env-Datei anlegen mit: ANTHROPIC_API_KEY=sk-ant-...

Ausführen:
    python claude_api_mc_benchmark.py

Achtung: Mit N=60 Iterationen und gemischten Prompt-Größen können je nach
Sampling ein paar Cent bis ein paar Dollar API-Kosten anfallen.
"""

import csv
import random
import statistics
import time
from dataclasses import asdict, dataclass, field

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
    "tiny":   10,     # ~13 Tokens
    "small":  100,    # ~130 Tokens
    "medium": 500,    # ~650 Tokens
    "large":  2000,   # ~2.6k Tokens
    "xlarge": 8000,   # ~10k Tokens
}

MAX_TOKENS_OPTIONS = [50, 200, 500, 1000]

# Basis-Wortpool, aus dem Prompts generiert werden
LOREM = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo "
    "consequat duis aute irure reprehenderit voluptate velit esse cillum"
).split()


def make_prompt(word_count: int, rng: random.Random) -> str:
    """Erzeugt einen Prompt mit ungefähr word_count Wörtern + klare Aufgabe."""
    words = (LOREM * (word_count // len(LOREM) + 1))[:word_count]
    rng.shuffle(words)
    return (
        " ".join(words)
        + "\n\nFasse den obigen Text in einem Satz auf Deutsch zusammen."
    )


# ----- Daten-Container -------------------------------------------------------

@dataclass
class Sample:
    iteration: int
    model: str
    prompt_size: str
    word_count: int
    max_tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    ttft: float = float("nan")     # Time To First Token (s)
    total: float = float("nan")    # Gesamt-Latenz (s)
    tps: float = 0.0               # Output-Tokens pro Sekunde
    error: str = ""


# ----- Einzelner API-Call ----------------------------------------------------

def single_run(it: int, model: str, size_name: str, word_count: int,
               max_tokens: int, rng: random.Random) -> Sample:
    prompt = make_prompt(word_count, rng)
    sample = Sample(
        iteration=it, model=model, prompt_size=size_name,
        word_count=word_count, max_tokens=max_tokens,
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

        s = single_run(i, model, size_name, word_count, max_tok, rng)
        samples.append(s)

        status = "ERR" if s.error else "OK "
        print(f"  [{i + 1:3d}/{n_iterations}] {status} {s.model:18s} "
              f"size={s.prompt_size:6s} max_tok={s.max_tokens:4d} "
              f"in={s.input_tokens:5d} out={s.output_tokens:4d} "
              f"total={s.total:6.2f}s ttft={s.ttft:5.2f}s tps={s.tps:5.1f}")

        # Jitter, um Rate Limits und Burst-Effekte zu glätten
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

    print("\nPro Prompt-Größe — TTFT (s):")
    for size in PROMPT_SIZES:
        print(_line(size, [s.ttft for s in ok if s.prompt_size == size]))

    if errs:
        print("\nFehler-Beispiele:")
        for e in errs[:5]:
            print(f"  - {e.model} {e.prompt_size}: {e.error}")


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
    ok = [s for s in samples if not s.error and s.ttft == s.ttft]  # NaN filtern
    if not ok:
        print("Keine erfolgreichen Samples zum Plotten.")
        return

    models_present = [m for m in MODELS if any(s.model == m for s in ok)]
    sizes_present = [sz for sz in PROMPT_SIZES if any(s.prompt_size == sz for s in ok)]
    color_map = {m: c for m, c in zip(MODELS, ["#4C72B0", "#DD8452", "#55A467"])}

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

    # 4) Scatter: Input-Tokens vs TTFT (Skalierung mit Input)
    ax = axes[1, 0]
    for m in models_present:
        xs = [s.input_tokens for s in ok if s.model == m]
        ys = [s.ttft for s in ok if s.model == m]
        ax.scatter(xs, ys, label=m, color=color_map[m], alpha=0.7, s=40)
    ax.set_xscale("log")
    ax.set_title("Input-Tokens → TTFT")
    ax.set_xlabel("Input-Tokens (log)")
    ax.set_ylabel("TTFT (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # 5) Scatter: Output-Tokens vs Total (Skalierung mit Output)
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

    # 6) Bar: Throughput (Tokens/s) pro Modell
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

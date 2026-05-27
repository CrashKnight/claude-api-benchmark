# Claude API Latency Benchmark

Monte-Carlo-Latenz-Benchmark für die Anthropic Claude API. Sampelt zufällige Kombinationen aus Modell, Prompt-Kategorie, Prompt-Größe und `max_tokens`, führt Streaming-API-Calls durch und berichtet Latenz-Verteilungen.

## Metriken

| Metrik | Beschreibung |
|--------|-------------|
| **TTFT** | Time To First Token — Zeit bis zum ersten generierten Token |
| **Total** | Gesamtlatenz des API-Calls |
| **TPS** | Output-Tokens pro Sekunde (Throughput) |

Alle Metriken werden als Mean, Median, P95, P99 und Standardabweichung ausgegeben — aufgeschlüsselt nach Modell, Kategorie und Prompt-Größe.

## Prompt-Kategorien

| Kategorie | Inhalt | Aufgabe |
|-----------|--------|---------|
| `prose` | Deutsche Wörter (shuffled) | Zusammenfassung in einem Satz |
| `code` | Generierter Python-Code | Code erklären |
| `json` | Verschachtelte JSON-Objekte | Alle Schlüssel auflisten |
| `instruction` | Nummerierte Aufgabenliste | Komplexesten Schritt benennen |

## Installation

```bash
pip install -r requirements.txt
```

## Konfiguration

`.env`-Datei im Projektverzeichnis anlegen:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Ausführen

```bash
python claude_api_mc_benchmark.py
```

Der Benchmark läuft standardmäßig **60 Iterationen** mit `seed=42`. Jede Iteration sampelt zufällig:
- **Modell:** `claude-haiku-4-5`, `claude-sonnet-4-5`, `claude-opus-4-5`
- **Prompt-Größe:** tiny (10 Wörter) bis xlarge (8.000 Wörter)
- **Kategorie:** prose, code, json, instruction
- **max_tokens:** 50, 200, 500 oder 1.000

## Output

Nach dem Durchlauf werden zwei Dateien erzeugt:

- `claude_api_mc_results.csv` — Rohdaten aller Samples
- `claude_api_mc_plots.png` — 6 Plots (Boxplots, Scatter, Throughput-Bars)

**Beispiel-Output (Konsole):**
```
  [  1/ 60] OK  claude-haiku-4-5   cat=prose       size=small  max_tok= 200 in=  143 out=  38 total=  1.12s ttft= 0.31s tps=189.4
  [  2/ 60] OK  claude-opus-4-5    cat=code        size=medium max_tok= 500 in=  672 out=  97 total=  3.84s ttft= 1.42s tps= 49.8
  ...
```

## Kosten

Mit N=60 Iterationen und gemischten Prompt-Größen entstehen je nach Sampling **einige Cent bis wenige Dollar** an API-Kosten.

## Erwartete Richtwerte

| Modell | TTFT (median) | Throughput |
|--------|--------------|------------|
| Haiku 4.5 | ~0.3s | ~195 tok/s |
| Sonnet 4.5 | ~0.65s | ~98 tok/s |
| Opus 4.5 | ~1.35s | ~52 tok/s |

*Werte variieren je nach Tageszeit und Serverlast.*

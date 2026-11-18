# PREREGISTRAZIONE — Goal notte 19 novembre 2026

**Obiettivo:** scegliere e migliorare il campione di intelligenza (Mellum 2.1 vs AREX-Turbo) su RTX 3060 12GB + 16GB RAM, Windows 11, llama.cpp b12141.

## Decisione a priori (prima dei risultati)
Campione = AREX-Turbo se E0 mostra AREX − Mellum ≥ +8 pp con ≥5 coppie discordanti. Altrimenti Mellum (vantaggio velocità + MTP).

## E0 — Confronto testuale appaiato (PREREQUISITO)
- **Ipotesi H1:** AREX ≥ Mellum di ≥8 pp sulle identiche domande.
- **Metrica:** accuratezza 0/1 per domanda (greedy), differenze accoppiate, kappa di Cohen, tasso di disaccordo.
- **Decision rule:** ≥+8 pp e ≥5 coppie discordanti → AREX primario; ≤+3 pp o Mellum vince → Mellum primario; 3-8 pp → inconcludente, n=50.
- **Budget:** 15 domande × 2 modelli × 1 greedy + 4 campioni T=0,7 × ~350 token. Stesso prompt, stessa normalizzazione, contenuto system allineato (template diversi).
- **Caveat dichiarato:** n=15 è DIREZIONALE, non robusto (±25 pp CI grezzo). Serve bootstrap 1000× e McNemar. Non si claiming "robustezza".

## E1 — Sweep temperatura SC (se AREX vince)
- **H2:** T=0,9 ottimizza SC; T=0,6 (default) è sottodiverso.
- **Metrica:** SC accuracy, diversità (distinte/k), greedy accuracy.
- **Decision rule:** adotta T se SC ≥ greedy +3 pp e diversità ≥0,4.
- **Budget:** 15 × 2 T × 4 sample × 350 token.

## E2 — Early-stop su SC (post-hoc, 0 generazione extra)
- **H3:** stop dopo 3 risposte uguali taglia ≥50% sample con ≤2 pp perdita.
- **Decision rule:** sample medi ≤4,5 e accuracy ≥ SC-8 − 2 pp.

## E3 — Gate greedy-pass (post-hoc, 0 generazione extra)
- **H4:** gate su margin/logprob del token-risposta salta ≥40% SC con ≤2 pp perdita.
- **Decision rule:** skip ≥40% e drop ≤2 pp. NON usare entropy media di catena.

## E4 — Voto pesato con confidenza (post-hoc)
- **H5:** voto pesato ≥ voto semplice a pari k.
- **Decision rule:** ≥+2 pp oppure stessa accuracy con ≤60% sample.

## E5 — Mellum MTP (già testato stanotte)
- Risultato: 52-62 t/s vs 100 baseline → **FLOP**, scartato. Il 1,6× di JetBrains è su H200; qui il draft da 8,3 GB satura la VRAM.

## E7 — Metodologia inference-optimisation-skill (aggiunta 19/11)
- **T1 drift:** ripetere E0-mellum-greedy con config identica. Se differisce >3 pp, i risultati vanno rivalutati (rumore dichiarato ±5-10%).
- **T3 KV dtype su AREX:** q4_0 (usato) vs q8_0. Prior sfavorevole (il repo riporta -41% su modelli con pochi layer KV), ma AREX è dense → misurare.
- **T6 quant AREX Q6_K:** stessa domande E0, verifica se maggiore precisione migliora il duro.
- **Leve TTFT (non decode):** --cache-ram 32768, --ctx-checkpoints 64, --checkpoint-every-n-tokens 4096, --slot-prompt-similarity 0.3, --cache-reuse 256. Guadagno atteso su prefissi ripetuti (38-53×), NON sul t/s single-user.
- **Non rifare:** MTP (net-negativo su MoE, 4 fonti), KV q4 come "win" (entro rumore), ngram su probe sintetici.

## Regole comuni
- Scrivere ipotesi/metrica/decision rule/budget PRIMA di ogni run (questo file).
- Logging in `results/windows/` con build.py; commit piccoli.
- Bootstrap 1000× e McNemar esatto per i confronti accoppiati.
- Niente claim con n=8 spacciato per robusto; differenze <3 pp = rumore.
- Niente distillazione con <1000 esempi (evidenza LIMO replication: −20 pp su 1,5B).

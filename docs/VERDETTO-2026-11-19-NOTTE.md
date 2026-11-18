# VERDETTO — Notte 19 novembre 2026 (lab factory)

Hardware: RTX 3060 12GB + 16GB DDR4 + Ryzen 5 5600G, Windows 11, llama.cpp b12141.
Metodo: `duro.py` seed 2026 (44 domande: 15 math, 15 code, 14 mmlu), `speedtest.py`, preregistrazione in `PREREG-2026-11-19-NOTTE.md`. n=44 è **direzionale**, non confermativo; differenze <3 pp = rumore. Drift gate: 3 run identici su Qwen35B (77,3% / 34/44 / 6,2-6,3 s) → misure stabili.

## Frontiera di Pareto (i due campioni)

| Campione | Duro | t/s | s/domanda | Quando usarlo |
|---|---|---|---|---|
| **Qwen3.8-27B UD-IQ2_XS** (denso, 7,1 GB) | **84,1%** (37/44) | 20,9 | 40,0 | **Qualità**: ragionamento, math, mmlu, task difficili |
| **Qwen3.8-35B-A3B-Distill IQ2_M** (MoE, 11,7 GB) | 77,3% (34/44) | **72,3** | **6,3** | **Velocità**: interattivo, agent loop, throughput |

Confronto completo (stesso set): Mellum 2.1 61,4% @100 t/s; AREX-Turbo 72,7% @71 t/s; Qwen35B 77,3% @72,3 t/s; Qwen27B 84,1% @20,9 t/s.

## Cosa ha funzionato
- **Qwen35B A3B** (su suggerimento utente): 72,3 t/s con IQ2_M tutto in VRAM, 77,3% duro, 6,3 s/domanda — batte AREX sia in qualità (+4,6 pp) sia in tempo/domanda (2,1× più rapido).
- **Qwen27B UD-IQ2_XS**: +6,8 pp di qualità rispetto a Qwen35B; math 86,7% vs 73,3%, mmlu 78,6% vs 71,4%.
- Confronto appaiato E0: AREX 72,7% vs Mellum 61,4% (+11,3 pp, reale per drift gate).

## Cosa è fallito (tutto registrato)
- **MTP su Mellum**: 52-62 t/s vs 100 baseline (draft da 8,3 GB satura VRAM) — confermato da 4 fonti community: net-negativo su MoE.
- **MTP su Qwen35B**: draft head IsValorum non compatibile col quant empero (crash in load).
- **IQ3_M** (15,2 GB): 16-21 t/s con qualsiasi offload (`-ngl 40`, `-ncmoe 36`, `-ncmoe 8`) — il PCIe annulla il guadagno qualità.
- **KV q4**: +1,7% = dentro il rumore.
- **Engine b12244**: 20 t/s (regressione -80%).
- **AREX Q6_K**: 50,0% vs Q4 54,2% su sottoinsieme (peggiora, entro rumore).
- **Self-consistency 4 voti su Qwen35B**: 69,0% vs 72,4% greedy e 3,5× più lenta — il distill soffre runaway thinking, il voto multiplo amplifica l'errore.
- **enable_thinking:false**: no-op (il distill genera già 423 token medi).
- **Alpha-Orionis-XS / Flash-Next su Strata Windows**: caricato ma NaN logit (formato Betelgeuse incompatibile).

## Lezioni metodologiche (inference-optimisation-skill)
- Skill di **procedura**, non libreria: repo di 3 giorni, 1065★, numeri self-reported e bug noti (ceiling.py conta due volte la concorrenza, mem.sh sbaglia unità 1024×). **Metodo valido, tabelle inutilizzabili**.
- Regole adottate: un solo server alla volta (RAM 16GB), controllo–candidato–controllo, mediane, ≥6 campioni, noise ±5-10%, drift gate, nessuna promozione da n<15.
- Siamo al ~90% del soffitto di banda della 3060 per il decode single-user: +10-30% di t/s puro è improbabile. I guadagni residui sono su qualità e TTFT/concorrenza.

## Verdetto operativo
- **Usa Qwen35B IQ2_M** per lavoro interattivo/agentico (72 t/s, 77% qualità, 6 s/domanda).
- **Usa Qwen27B UD-IQ2_XS** quando serve la massima qualità (84% duro), accettando 40 s/domanda.
- Non esiste un modello che batta entrambi simultaneamente su questa hardware: la scelta è Pareto, non assoluta.
- Prossimo fronte (se continua): engine alternativi (ik_llama.cpp, Strata/RustingStrata) sulla stessa architettura 35B-A3B per provare a sbloccare 100+ t/s senza perdere qualità; pesi finali empero v2 in training.

Repo: 291 esperimenti, 0 errori. Ultimo commit: E12.

# Transfer Learning für Luftqualitätsprognosen mit einem Time Series Foundation Model

**Semesterarbeit Deep Learning · FH Südwestfalen (Betreuung: Prof. Dr. Stefan Goetze) · WiSe 2025/26**
Autor: Simeon Ehmer · Lizenz: [CC BY-SA 4.0](LICENSE)

## Problem

Klassische Prognosemodelle für Luftqualität sind stationsgebunden: Jede neue Messstation braucht 1–2 Jahre lokale Daten, bevor ein Modell Jahreszeiten, Ferien und Wetterlagen gelernt hat — und ein in Duisburg trainiertes Modell ist am nächsten Standort wertlos. Für Klimaquartiere heißt das: Die Wirkungsbilanz kommt oft erst nach der Förderperiode.

## Ansatz

Ein offenes Time-Series-Foundation-Model — [MOMENT](https://github.com/moment-timeseries-foundation-model/moment) (Carnegie Mellon, ICML 2024, MIT-Lizenz, 385 Mio. Parameter) — wird dreistufig angepasst:

1. **Pre-Training** auf 18 LANUK-Stationen (10,85 Mio. Datenpunkte, selbstüberwacht)
2. **Fine-Tuning** auf 13 Stationen: 24-h-Prognose des WCAQI (gewichteter Luftqualitätsindex aus PM2.5 · PM10 · NO₂ · O₃ · NO, normiert auf EU-Grenzwerte) — Val-R² 0,80
3. **Transfer Learning** auf drei vom gesamten Training ausgeschlossene Stationen: Dortmund-Eving (industriell), Schwerte (suburban), Düsseldorf-Lörick (urban)

## Ergebnis

| Szenario | Ø R² (3 ungesehene Stationen) |
|---|---|
| **Zero-Shot** (0 lokale Datenpunkte, Tag 1) | **0,676** |
| Few-Shot 10 % (≈ 2 Wochen Daten), LR 1e-6 | 0,703 |
| Few-Shot 25 % (≈ 1 Monat), LR 1e-6 | 0,719 |
| Full Fine-Tuning, LR 1e-6 | 0,729 (bester Einzelwert 0,863) |

Skill-Score gegen Persistence: 0,52–0,61 · Trendrichtung in 65–68 % der Fälle korrekt.

**Nebenbefund (Catastrophic Forgetting):** Mit Standard-Lernrate (5e-5) machten lokale Daten das Modell *schlechter* als Zero-Shot. Eine 50-fach kleinere Lernrate (1e-6) eliminierte das Vergessen vollständig; Backbone-Einfrieren half dagegen nicht.

**Praktische Bedeutung:** Ein neuer Sensor liefert ab der ersten Stunde Prognosen — Wirkungsbilanz ab Inbetriebnahme statt nach Jahren. Jeder weitere Standort ist in Tagen angebunden; dasselbe Modellartefakt (1,34 GB) ist unverändert auf jede Kommune mit WCAQI-Zeitreihen übertragbar.

## Grenzen (ehrlich)

R² ≈ 0,68 (Zero-Shot) ist eine **informative Entscheidungshilfe**, keine regulatorisch belastbare Prognose im Sinne der EU-AAQD 2024/2881. MOMENT liefert Punktprognosen ohne Konfidenzintervalle — Conformal Prediction ist in Vorbereitung.

## Daten

**Es werden keine Rohdaten in diesem Repository verteilt.** Alle Messdaten stammen aus dem offenen Messnetz des **Landesamts für Natur, Umwelt und Klima NRW (LANUK, vormals LANUV)**, Zeitraum 2020–2024, 18 Stationen, stündliche Werte:

- Open-Data-API: https://luftqualitaet.nrw.de/apidoku.php
- Die Vorverarbeitungs-Schritte (Ausreißer, Lücken, WCAQI-Berechnung) sind im Notebook dokumentiert und reproduzierbar.

## Inhalt

- `DeepLearning_Transfer_Learning_Luftqualitaet.ipynb` — vollständiges Notebook (Methodik, Experimente, Ergebnisse, inkl. der Irrwege). Hinweis: Ergebnis-Abbildungen referenzieren lokale Läufe und entstehen beim Ausführen; sie sind nicht Teil des Repositories.

## Verwandte Arbeiten

- [ML-LANUK-Luftqualitaet](https://github.com/Simeon1970-prog/ML-LANUK-Luftqualitaet) — Vorgängerprojekt mit klassischem ML (XGBoost/Ensemble, R² 0,87 mit / 0,13 ohne Lag-Features)
- [fehlende-werte-luftqualitaet](https://github.com/Simeon1970-prog/fehlende-werte-luftqualitaet) — Begleit-Notebook zu fehlenden Werten (MCAR/MAR/MNAR)
- [Positive-Climate-District](https://github.com/Simeon1970-prog/Positive-Climate-District) — Framework, in dem diese Prognosefähigkeit zur Wirkungsmessung von Klimaquartieren eingesetzt wird

## Referenzen

- Goswami et al. (2024): *MOMENT: A Family of Open Time-series Foundation Models*, ICML 2024 — https://arxiv.org/abs/2402.03885
- Modellgewichte: https://huggingface.co/AutonLab/MOMENT-1-large (MIT)
- Pan & Yang (2010): *A Survey on Transfer Learning* — https://doi.org/10.1109/TKDE.2009.191
- Iskandaryan, Ramos & Trilles (2020), *Applied Sciences* 10(7), 2401 — https://doi.org/10.3390/app10072401

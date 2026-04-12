# Evidence Ledger

> Topic:
> Last updated: YYYY-MM-DD

## Ledger

| ID | Bucket | Source | Date | Type | Question IDs | Hypotheses (supports/contradicts) | Evidence type | Strength | Setting / conditions | Fetched | Verified | Key findings / quantitative result | Caveats / limitations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | Direct evidence | [Title](URL) | YYYY-MM-DD | paper / docs / benchmark / repo | Q1 | +H1, -H2 | direct | strong | dataset=..., model=..., setting=... | yes | yes |  |  |
| E2 | Counterevidence / caveat | [Title](URL) | YYYY-MM-DD | paper | Q1 | -H1 | direct | moderate | condition=... | yes | yes |  |  |

## Notes

- Use `Bucket` to group sources: direct evidence, mechanistic evidence, mitigation evidence, counterevidence/caveats, surveys for orientation.
- Use `Evidence type` to distinguish **direct**, **adjacent**, or **framing** evidence.
- Keep `Fetched` and `Verified` explicit (yes/no). If only partially checked, set `Verified=no` and describe the partial check in notes.
- Record conditions/settings whenever possible; many contradictions disappear when conditions differ.

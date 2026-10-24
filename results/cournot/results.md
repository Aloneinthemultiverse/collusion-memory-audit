# Results

## Controls (last 10 rounds)

| Condition | Division index | Profit vs Nash |
|---|---|---|
| C0_no_cross_agent | 0.87 ± 0.13 (n=2) | -55.60 ± 27.60 (n=2) % |
| C1_all_channels | 0.49 ± 0.47 (n=13) | -35.26 ± 24.20 (n=13) % |

## Audit cost (replays until first verified pathway; effect ≥ 0.3)

| game | n | verified | random (expected) | heuristic | random (1 draw) | laya |
|---|---|---|---|---|---|---|
| seed0 | 6 | 1 | 3.50 | 6 | 5 | 6 |
| seed12 | 6 | 0 | None | None | None | None |
| seed3 | 6 | 3 | 1.75 | 1 | 2 | 2 |
| seed4 | 6 | 3 | 1.75 | 2 | 4 | 1 |
| seed9 | 6 | 2 | 2.33 | 1 | 1 | 1 |

## Candidate effects (null-replay division − masked-replay division)

| Game | Candidate | Effect | Verified |
|---|---|---|---|
| seed0 | rival_history:Firm A->Firm B@10 | +0.00 | no |
| seed0 | rival_history:Firm B->Firm A@10 | +0.00 | no |
| seed0 | board:Firm A->Firm B@10 | +0.00 | no |
| seed0 | board:Firm B->Firm A@10 | +0.00 | no |
| seed0 | memory:Firm A->Firm A@10 | +0.00 | no |
| seed0 | memory:Firm B->Firm B@10 | +0.52 | yes |
| seed12 | rival_history:Firm A->Firm B@10 | +0.01 | no |
| seed12 | rival_history:Firm B->Firm A@10 | +0.00 | no |
| seed12 | board:Firm A->Firm B@10 | +0.10 | no |
| seed12 | board:Firm B->Firm A@10 | -0.20 | no |
| seed12 | memory:Firm A->Firm A@10 | -0.02 | no |
| seed12 | memory:Firm B->Firm B@10 | -0.20 | no |
| seed3 | rival_history:Firm A->Firm B@10 | -0.10 | no |
| seed3 | rival_history:Firm B->Firm A@10 | +0.43 | yes |
| seed3 | board:Firm A->Firm B@10 | -0.10 | no |
| seed3 | board:Firm B->Firm A@10 | -0.10 | no |
| seed3 | memory:Firm A->Firm A@10 | +0.33 | yes |
| seed3 | memory:Firm B->Firm B@10 | +0.57 | yes |
| seed4 | rival_history:Firm A->Firm B@10 | -0.03 | no |
| seed4 | rival_history:Firm B->Firm A@10 | -0.03 | no |
| seed4 | board:Firm A->Firm B@10 | +0.48 | yes |
| seed4 | board:Firm B->Firm A@10 | +0.37 | yes |
| seed4 | memory:Firm A->Firm A@10 | +0.01 | no |
| seed4 | memory:Firm B->Firm B@10 | +0.31 | yes |
| seed9 | rival_history:Firm A->Firm B@10 | +0.50 | yes |
| seed9 | rival_history:Firm B->Firm A@10 | +0.51 | yes |
| seed9 | board:Firm A->Firm B@10 | -0.15 | no |
| seed9 | board:Firm B->Firm A@10 | -0.13 | no |
| seed9 | memory:Firm A->Firm A@10 | -0.15 | no |
| seed9 | memory:Firm B->Firm B@10 | +0.30 | no |

## Minimal cut (channel removed for both firms from round 1)

| Game | Removed | Division | Effect |
|---|---|---|---|
| seed0 | rival_history | 0.96 | -0.78 |
| seed0 | board | 0.34 | -0.17 |
| seed0 | memory | 0.82 | -0.65 |
| seed0 | all | 0.38 | -0.21 |
| seed12 | rival_history | 0.00 | +0.78 |
| seed12 | board | 0.56 | +0.22 |
| seed12 | memory | 0.47 | +0.30 |
| seed12 | all | 0.05 | +0.72 |
| seed3 | rival_history | 0.52 | -0.49 |
| seed3 | board | 0.31 | -0.27 |
| seed3 | memory | 0.57 | -0.54 |
| seed3 | all | 0.29 | -0.25 |
| seed4 | rival_history | 0.57 | -0.36 |
| seed4 | board | 0.85 | -0.64 |
| seed4 | memory | 0.42 | -0.21 |
| seed4 | all | 0.51 | -0.30 |
| seed9 | rival_history | 0.00 | +0.49 |
| seed9 | board | 0.83 | -0.35 |
| seed9 | memory | 0.57 | -0.08 |
| seed9 | all | 0.31 | +0.18 |
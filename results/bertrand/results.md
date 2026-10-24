# Results

## Controls (last 10 rounds)

| Condition | Division index | Profit vs Nash |
|---|---|---|
| C0_no_cross_agent | 0.60 ± 0.47 (n=5) | 23.27 ± 23.67 (n=5) % |
| C1_all_channels | 0.31 ± 0.47 (n=5) | 14.95 ± 21.66 (n=5) % |

## Audit cost (replays until first verified pathway; effect ≥ 0.3)

| game | n | verified | random (expected) | heuristic | random (1 draw) | laya |
|---|---|---|---|---|---|---|
| seed1 | 6 | 0 | None | None | None | None |

## Candidate effects (null-replay division − masked-replay division)

| Game | Candidate | Effect | Verified |
|---|---|---|---|
| seed1 | rival_history:Firm A->Firm B@10 | +0.02 | no |
| seed1 | rival_history:Firm B->Firm A@10 | +0.02 | no |
| seed1 | board:Firm A->Firm B@10 | +0.06 | no |
| seed1 | board:Firm B->Firm A@10 | -0.03 | no |
| seed1 | memory:Firm A->Firm A@10 | -0.03 | no |
| seed1 | memory:Firm B->Firm B@10 | +0.26 | no |

## Minimal cut (channel removed for both firms from round 1)

| Game | Removed | Division | Effect |
|---|---|---|---|
| seed1 | rival_history | 0.90 | -0.52 |
| seed1 | board | -0.04 | +0.41 |
| seed1 | memory | 0.64 | -0.26 |
| seed1 | all | 1.49 | -1.11 |
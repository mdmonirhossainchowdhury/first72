# First72 synthetic benchmark (30 scenes per difficulty)

> Synthetic scenes with exact ground truth. These numbers validate the pipeline and compare methods; they are **not** real-world accuracy. Re-run on real tiles with `python -m first72 evaluate`.

## Easy scenes

| Method | IoU | Precision | Recall | F1 | ECE (lower=better) | Top-5 zone overlap | Interval coverage |
|---|---|---|---|---|---|---|---|
| Post-only water threshold (baseline) | 0.831 ± 0.026 | 0.833 ± 0.026 | 0.996 ± 0.002 | 0.907 ± 0.016 | 0.071 ± 0.006 | 0.887 ± 0.112 | 1.000 ± 0.000 |
| Pre/post change detection | 0.987 ± 0.003 | 0.995 ± 0.002 | 0.992 ± 0.003 | 0.993 ± 0.002 | 0.041 ± 0.005 | 0.933 ± 0.107 | 1.000 ± 0.000 |
| Change detection + audit agent (First72) | 0.987 ± 0.003 | 0.995 ± 0.002 | 0.992 ± 0.003 | 0.993 ± 0.002 | 0.041 ± 0.005 | 0.933 ± 0.107 | 1.000 ± 0.000 |

Does the agent's verdict tell you where the map is wrong? Mean zone-level IoU vs ground truth of the audited zones, by verdict:

| Verdict | Zones | Mean zone IoU |
|---|---|---|
| confirmed | 240 | 0.990 |
| revised | 0 | n/a |
| unresolved | 0 | n/a |

## Hard scenes (muddy water, heavy haze/shadow, more noise)

| Method | IoU | Precision | Recall | F1 | ECE (lower=better) | Top-5 zone overlap | Interval coverage |
|---|---|---|---|---|---|---|---|
| Post-only water threshold (baseline) | 0.640 ± 0.131 | 0.643 ± 0.134 | 0.993 ± 0.008 | 0.772 ± 0.101 | 0.172 ± 0.078 | 0.773 ± 0.184 | 0.876 ± 0.180 |
| Pre/post change detection | 0.587 ± 0.103 | 0.951 ± 0.048 | 0.604 ± 0.100 | 0.734 ± 0.083 | 0.081 ± 0.032 | 0.540 ± 0.194 | 0.715 ± 0.108 |
| Change detection + audit agent (First72) | 0.590 ± 0.106 | 0.951 ± 0.049 | 0.607 ± 0.103 | 0.736 ± 0.086 | 0.080 ± 0.033 | 0.527 ± 0.175 | 0.716 ± 0.109 |

Does the agent's verdict tell you where the map is wrong? Mean zone-level IoU vs ground truth of the audited zones, by verdict:

| Verdict | Zones | Mean zone IoU |
|---|---|---|
| confirmed | 103 | 0.830 |
| revised | 9 | 0.924 |
| unresolved | 128 | 0.462 |


# Same-information appended baseline results

All numbers refer to the already consumed 2,000-scene confirmation bank; not new confirmation.

| Track | Arm | Accepted | Unsafe accepted | Safe accepted | Recovered | Newly lost | New unsafe | Eta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| target | uniform | 1540 | 2 | 1538 | 0 | 0 | 0 | 96.6080% |
| target | shift | 1549 | 5 | 1544 | 6 | 0 | 3 | 96.9849% |
| target | plic | 1600 | 18 | 1582 | 44 | 0 | 16 | 99.3719% |
| learned | uniform | 7783 | 80 | 7703 | 0 | 0 | 0 | 96.7714% |
| learned | shift | 7783 | 80 | 7703 | 0 | 0 | 0 | 96.7714% |
| learned | plic | 7909 | 123 | 7786 | 83 | 0 | 43 | 97.8141% |

Target counts are over 2,000 scenes. Learned counts sum five models over those same scenes, not 10,000 independent trials.
Global shift: target b=.001, learned b=0. It was selected with no increase in development unsafe count. PLIC introduces additional danger; recovery alone is not overall superiority.
All new and resolved events and their exact signed decompositions are retained in EVENTS.jsonl.

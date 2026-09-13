# Full-family empirical risk comparison

Both banks were already consumed. Thresholds are outcome-selected within each bank; these are descriptive empirical envelopes, not certificates or out-of-sample rules. Zero acceptance has undefined risk. All equal scores enter together.

## fixed, development

| Track | Risk ceiling | Uniform eta | PLIC eta | Delta (pp) | Pointwise descriptive 95% interval (pp) |
|---|---:|---:|---:|---:|---|
| target | 0.5% | 97.880% | 99.501% | +1.621 | [+0.885, +2.668] |
| target | 1.0% | 98.317% | 99.751% | +1.434 | [+0.314, +2.189] |
| target | 1.5% | 99.190% | 99.813% | +0.623 | [+0.062, +1.565] |
| target | 2.0% | 99.501% | 99.938% | +0.436 | [+0.000, +0.944] |
| target | 2.5% | 99.813% | 99.938% | +0.125 | [+0.000, +0.578] |
| target | 3.0% | 99.813% | 100.000% | +0.187 | [+0.000, +0.377] |
| target | 4.0% | 99.813% | 100.000% | +0.187 | [+0.000, +0.433] |
| target | 5.0% | 99.813% | 100.000% | +0.187 | [+0.000, +0.378] |
| learned | 0.5% | 95.012% | 95.611% | +0.599 | [-0.485, +1.726] |
| learned | 1.0% | 96.796% | 97.419% | +0.623 | [-0.174, +1.531] |
| learned | 1.5% | 97.893% | 98.367% | +0.474 | [-0.099, +0.980] |
| learned | 2.0% | 98.579% | 98.778% | +0.200 | [-0.139, +0.651] |
| learned | 2.5% | 99.015% | 99.090% | +0.075 | [-0.161, +0.482] |
| learned | 3.0% | 99.264% | 99.327% | +0.062 | [-0.162, +0.350] |
| learned | 4.0% | 99.551% | 99.663% | +0.112 | [-0.037, +0.260] |
| learned | 5.0% | 99.688% | 99.726% | +0.037 | [-0.024, +0.213] |

## fixed, appended

| Track | Risk ceiling | Uniform eta | PLIC eta | Delta (pp) | Pointwise descriptive 95% interval (pp) |
|---|---:|---:|---:|---:|---|
| target | 0.5% | 97.299% | 98.807% | +1.508 | [+0.520, +2.527] |
| target | 1.0% | 98.430% | 99.309% | +0.879 | [+0.311, +1.820] |
| target | 1.5% | 98.744% | 99.560% | +0.817 | [+0.249, +1.322] |
| target | 2.0% | 99.058% | 99.623% | +0.565 | [+0.063, +1.173] |
| target | 2.5% | 99.309% | 99.749% | +0.440 | [+0.000, +0.940] |
| target | 3.0% | 99.372% | 99.874% | +0.503 | [+0.126, +0.878] |
| target | 4.0% | 99.623% | 99.874% | +0.251 | [+0.062, +0.704] |
| target | 5.0% | 99.686% | 99.874% | +0.188 | [+0.000, +0.444] |
| learned | 0.5% | 94.535% | 94.987% | +0.452 | [-0.482, +2.060] |
| learned | 1.0% | 96.658% | 97.111% | +0.452 | [-0.127, +1.488] |
| learned | 1.5% | 97.211% | 97.751% | +0.540 | [-0.039, +0.844] |
| learned | 2.0% | 97.827% | 98.128% | +0.302 | [-0.187, +0.662] |
| learned | 2.5% | 98.417% | 98.430% | +0.013 | [-0.248, +0.453] |
| learned | 3.0% | 98.706% | 98.832% | +0.126 | [-0.215, +0.503] |
| learned | 4.0% | 99.183% | 99.372% | +0.188 | [-0.012, +0.476] |
| learned | 5.0% | 99.359% | 99.535% | +0.176 | [+0.000, +0.379] |

## reselected, development

| Track | Risk ceiling | Uniform eta | PLIC eta | Delta (pp) | Pointwise descriptive 95% interval (pp) |
|---|---:|---:|---:|---:|---|
| target | 0.5% | 97.880% | 99.501% | +1.621 | [+0.907, +2.668] |
| target | 1.0% | 98.317% | 99.751% | +1.434 | [+0.314, +2.189] |
| target | 1.5% | 99.190% | 99.813% | +0.623 | [+0.062, +1.565] |
| target | 2.0% | 99.501% | 99.938% | +0.436 | [+0.000, +0.944] |
| target | 2.5% | 99.813% | 99.938% | +0.125 | [+0.000, +0.578] |
| target | 3.0% | 99.813% | 100.000% | +0.187 | [+0.000, +0.377] |
| target | 4.0% | 99.813% | 100.000% | +0.187 | [+0.000, +0.433] |
| target | 5.0% | 99.813% | 100.000% | +0.187 | [+0.000, +0.378] |
| learned | 0.5% | 95.012% | 95.536% | +0.524 | [-0.482, +1.740] |
| learned | 1.0% | 96.796% | 97.444% | +0.648 | [-0.163, +1.626] |
| learned | 1.5% | 97.893% | 98.354% | +0.461 | [-0.113, +0.980] |
| learned | 2.0% | 98.579% | 98.791% | +0.212 | [-0.134, +0.675] |
| learned | 2.5% | 99.015% | 99.102% | +0.087 | [-0.161, +0.509] |
| learned | 3.0% | 99.264% | 99.339% | +0.075 | [-0.161, +0.390] |
| learned | 4.0% | 99.551% | 99.676% | +0.125 | [-0.025, +0.289] |
| learned | 5.0% | 99.688% | 99.738% | +0.050 | [-0.012, +0.258] |

## reselected, appended

| Track | Risk ceiling | Uniform eta | PLIC eta | Delta (pp) | Pointwise descriptive 95% interval (pp) |
|---|---:|---:|---:|---:|---|
| target | 0.5% | 97.299% | 98.807% | +1.508 | [+0.562, +2.563] |
| target | 1.0% | 98.430% | 99.309% | +0.879 | [+0.311, +1.820] |
| target | 1.5% | 98.744% | 99.560% | +0.817 | [+0.249, +1.322] |
| target | 2.0% | 99.058% | 99.623% | +0.565 | [+0.063, +1.173] |
| target | 2.5% | 99.309% | 99.623% | +0.314 | [+0.000, +0.940] |
| target | 3.0% | 99.372% | 99.874% | +0.503 | [+0.124, +0.877] |
| target | 4.0% | 99.623% | 99.874% | +0.251 | [+0.062, +0.704] |
| target | 5.0% | 99.686% | 99.874% | +0.188 | [+0.000, +0.444] |
| learned | 0.5% | 94.535% | 94.925% | +0.389 | [-0.511, +2.046] |
| learned | 1.0% | 96.658% | 97.123% | +0.465 | [-0.153, +1.406] |
| learned | 1.5% | 97.211% | 97.751% | +0.540 | [-0.038, +0.843] |
| learned | 2.0% | 97.827% | 98.128% | +0.302 | [-0.150, +0.670] |
| learned | 2.5% | 98.417% | 98.442% | +0.025 | [-0.212, +0.513] |
| learned | 3.0% | 98.706% | 98.844% | +0.138 | [-0.215, +0.525] |
| learned | 4.0% | 99.183% | 99.397% | +0.214 | [-0.012, +0.524] |
| learned | 5.0% | 99.359% | 99.598% | +0.239 | [+0.025, +0.465] |

## Development-selected thresholds on consumed appended bank

These thresholds are selected only on development and then applied unchanged; actual appended risk is reported and need not meet the development ceiling.

| Selector | Track | Development ceiling | Arm | Threshold | Actual appended risk | Appended eta |
|---|---|---:|---|---:|---:|---:|
| fixed | target | 0.5% | uniform | 0.022987883667790716 | 0.638% | 97.802% |
| fixed | target | 1.0% | uniform | 0.02479757085020243 | 0.886% | 98.367% |
| fixed | target | 1.5% | uniform | 0.028004177939111675 | 1.565% | 98.744% |
| fixed | target | 2.0% | uniform | 0.029498550056913653 | 1.930% | 98.932% |
| fixed | target | 2.5% | uniform | 0.031625613678686305 | 2.409% | 99.246% |
| fixed | target | 3.0% | uniform | 0.031625613678686305 | 2.409% | 99.246% |
| fixed | target | 4.0% | uniform | 0.031625613678686305 | 2.409% | 99.246% |
| fixed | target | 5.0% | uniform | 0.031625613678686305 | 2.409% | 99.246% |
| fixed | target | 0.5% | plic | 0.01785025070576833 | 0.629% | 99.246% |
| fixed | target | 1.0% | plic | 0.019396095174022458 | 1.002% | 99.309% |
| fixed | target | 1.5% | plic | 0.02128053594296372 | 1.613% | 99.623% |
| fixed | target | 2.0% | plic | 0.022595655460395964 | 2.219% | 99.623% |
| fixed | target | 2.5% | plic | 0.022595655460395964 | 2.219% | 99.623% |
| fixed | target | 3.0% | plic | 0.026971481051102652 | 3.402% | 99.874% |
| fixed | target | 4.0% | plic | 0.026971481051102652 | 3.402% | 99.874% |
| fixed | target | 5.0% | plic | 0.026971481051102652 | 3.402% | 99.874% |
| fixed | learned | 0.5% | uniform | 0.015677556617411888 | 0.579% | 94.874% |
| fixed | learned | 1.0% | uniform | 0.019690441724881096 | 1.003% | 96.671% |
| fixed | learned | 1.5% | uniform | 0.02303429509629854 | 1.623% | 97.450% |
| fixed | learned | 2.0% | uniform | 0.02585832501022187 | 2.156% | 98.065% |
| fixed | learned | 2.5% | uniform | 0.028207772889929213 | 2.514% | 98.417% |
| fixed | learned | 3.0% | uniform | 0.0302016385834813 | 2.964% | 98.706% |
| fixed | learned | 4.0% | uniform | 0.0345256160362911 | 3.956% | 99.133% |
| fixed | learned | 5.0% | uniform | 0.039060606448370276 | 5.167% | 99.384% |
| fixed | learned | 0.5% | plic | 0.012677813357918614 | 0.563% | 95.352% |
| fixed | learned | 1.0% | plic | 0.01707690634753977 | 1.112% | 97.173% |
| fixed | learned | 1.5% | plic | 0.02078162885465065 | 1.702% | 97.940% |
| fixed | learned | 2.0% | plic | 0.022918598221819984 | 2.165% | 98.229% |
| fixed | learned | 2.5% | plic | 0.02515395618195938 | 2.644% | 98.518% |
| fixed | learned | 3.0% | plic | 0.02716230833719419 | 2.973% | 98.819% |
| fixed | learned | 4.0% | plic | 0.032349293466248825 | 4.086% | 99.372% |
| fixed | learned | 5.0% | plic | 0.03619225523798841 | 4.956% | 99.510% |
| reselected | target | 0.5% | uniform | 0.022987883667790716 | 0.638% | 97.802% |
| reselected | target | 1.0% | uniform | 0.02479757085020243 | 0.886% | 98.367% |
| reselected | target | 1.5% | uniform | 0.028004177939111675 | 1.565% | 98.744% |
| reselected | target | 2.0% | uniform | 0.029498550056913653 | 1.930% | 98.932% |
| reselected | target | 2.5% | uniform | 0.031625613678686305 | 2.409% | 99.246% |
| reselected | target | 3.0% | uniform | 0.031625613678686305 | 2.409% | 99.246% |
| reselected | target | 4.0% | uniform | 0.031625613678686305 | 2.409% | 99.246% |
| reselected | target | 5.0% | uniform | 0.031625613678686305 | 2.409% | 99.246% |
| reselected | target | 0.5% | plic | 0.01785025070576833 | 0.629% | 99.246% |
| reselected | target | 1.0% | plic | 0.019396095174022458 | 1.002% | 99.309% |
| reselected | target | 1.5% | plic | 0.02128053594296372 | 1.613% | 99.623% |
| reselected | target | 2.0% | plic | 0.022595655460395964 | 2.280% | 99.623% |
| reselected | target | 2.5% | plic | 0.022595655460395964 | 2.280% | 99.623% |
| reselected | target | 3.0% | plic | 0.026971481051102652 | 3.402% | 99.874% |
| reselected | target | 4.0% | plic | 0.026971481051102652 | 3.402% | 99.874% |
| reselected | target | 5.0% | plic | 0.026971481051102652 | 3.402% | 99.874% |
| reselected | learned | 0.5% | uniform | 0.015677556617411888 | 0.579% | 94.874% |
| reselected | learned | 1.0% | uniform | 0.019690441724881096 | 1.003% | 96.671% |
| reselected | learned | 1.5% | uniform | 0.02303429509629854 | 1.623% | 97.450% |
| reselected | learned | 2.0% | uniform | 0.02585832501022187 | 2.156% | 98.065% |
| reselected | learned | 2.5% | uniform | 0.028207772889929213 | 2.514% | 98.417% |
| reselected | learned | 3.0% | uniform | 0.0302016385834813 | 2.964% | 98.706% |
| reselected | learned | 4.0% | uniform | 0.0345256160362911 | 3.956% | 99.133% |
| reselected | learned | 5.0% | uniform | 0.039060606448370276 | 5.167% | 99.384% |
| reselected | learned | 0.5% | plic | 0.012451162199956997 | 0.564% | 95.302% |
| reselected | learned | 1.0% | plic | 0.01707690634753977 | 1.112% | 97.186% |
| reselected | learned | 1.5% | plic | 0.020639803372801843 | 1.665% | 97.927% |
| reselected | learned | 2.0% | plic | 0.022918598221819984 | 2.164% | 98.241% |
| reselected | learned | 2.5% | plic | 0.02515395618195938 | 2.644% | 98.543% |
| reselected | learned | 3.0% | plic | 0.02716230833719419 | 2.972% | 98.832% |
| reselected | learned | 4.0% | plic | 0.032349293466248825 | 4.096% | 99.410% |
| reselected | learned | 5.0% | plic | 0.03619225523798841 | 4.908% | 99.560% |

Continuation decision: **STOP_NO_NEW_SCENES**.
Positive point estimates do not establish equivalence, inferiority, or a stable advantage. No new scene is generated when the fixed screen fails.

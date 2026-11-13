# Changelog

## [0.3.1](https://github.com/gabrielkpo/hpc-inference-sizing/compare/v0.3.0...v0.3.1) (2026-11-13)


### Documentation

* mark the project as done in the README ([#33](https://github.com/gabrielkpo/hpc-inference-sizing/issues/33)) ([76a5144](https://github.com/gabrielkpo/hpc-inference-sizing/commit/76a5144543c41f1a1bb4095ff3aa2e15a02d7b25))

## [0.3.0](https://github.com/gabrielkpo/hpc-inference-sizing/compare/v0.2.0...v0.3.0) (2026-11-13)


### Features

* alert first on a stall or a campaign that never starts, destroy after a grace period ([#20](https://github.com/gabrielkpo/hpc-inference-sizing/issues/20)) ([e2170cd](https://github.com/gabrielkpo/hpc-inference-sizing/commit/e2170cd3d69fa031d5af10cd8a486550f93de2d4))
* chain the levers, the best combination and the repeats after the base matrix ([#18](https://github.com/gabrielkpo/hpc-inference-sizing/issues/18)) ([618ab9b](https://github.com/gabrielkpo/hpc-inference-sizing/commit/618ab9b25e6991bfc88eef31c6cd9208fac4e26d))
* destroy the node when the campaign never starts or stalls ([#19](https://github.com/gabrielkpo/hpc-inference-sizing/issues/19)) ([44430d1](https://github.com/gabrielkpo/hpc-inference-sizing/commit/44430d17441ffddc29af6937d1e35980e71006fa))
* follow a long campaign from the laptop and end the rental when it is done ([#16](https://github.com/gabrielkpo/hpc-inference-sizing/issues/16)) ([75f722a](https://github.com/gabrielkpo/hpc-inference-sizing/commit/75f722a68d6af930477848f7000e9156e8412d2c))
* prepare the 4-GPU session: two units behind nginx and PIAF quality runs ([#30](https://github.com/gabrielkpo/hpc-inference-sizing/issues/30)) ([a27ee27](https://github.com/gabrielkpo/hpc-inference-sizing/commit/a27ee27b310ac7d970dd14cfe4bd171638ad90cb))
* sizing tool answering both clients from the measured capacities ([#28](https://github.com/gabrielkpo/hpc-inference-sizing/issues/28)) ([74beae8](https://github.com/gabrielkpo/hpc-inference-sizing/commit/74beae8114802de9956369437c80ec7ac89141e6))
* summarize the campaign's run files and publish the 2-GPU results ([#26](https://github.com/gabrielkpo/hpc-inference-sizing/issues/26)) ([a61f66a](https://github.com/gabrielkpo/hpc-inference-sizing/commit/a61f66aa7392f6514134c4c3687a1ad81bd56250))
* unmeasured warm-up and automatic provenance in every run ([#29](https://github.com/gabrielkpo/hpc-inference-sizing/issues/29)) ([b5098ba](https://github.com/gabrielkpo/hpc-inference-sizing/commit/b5098bae8030e636885ebf05469aa7dbc97ad5e5))


### Bug Fixes

* fetch only the Mistral-format weights in the node setup ([#32](https://github.com/gabrielkpo/hpc-inference-sizing/issues/32)) ([c0eba3b](https://github.com/gabrielkpo/hpc-inference-sizing/commit/c0eba3bd5fe0a1213b14fb6233a6d95bbd3a55a7))
* never measure a leftover server under a new unit's name ([#24](https://github.com/gabrielkpo/hpc-inference-sizing/issues/24)) ([b86b0d3](https://github.com/gabrielkpo/hpc-inference-sizing/commit/b86b0d33c459f63f5f93c786ca28e23a536a1893))
* read each target's capacity from every measured step and record the rate actually sent ([#22](https://github.com/gabrielkpo/hpc-inference-sizing/issues/22)) ([a3fe741](https://github.com/gabrielkpo/hpc-inference-sizing/commit/a3fe74103a18b4f10e3a0ece3e7855f9977b6257))
* take only expected files from the node and keep events out of its reach ([#23](https://github.com/gabrielkpo/hpc-inference-sizing/issues/23)) ([539440a](https://github.com/gabrielkpo/hpc-inference-sizing/commit/539440ad8c40bd2ba72ebc586b8a99f21c445c43))


### Documentation

* add a plain-language page for decision-makers to the results ([#27](https://github.com/gabrielkpo/hpc-inference-sizing/issues/27)) ([69c29d6](https://github.com/gabrielkpo/hpc-inference-sizing/commit/69c29d6e57e79e87c70d3356e6f59bd907a52c82))
* publish the 4-GPU session: two units behind a balancer and NVFP4 quality ([#31](https://github.com/gabrielkpo/hpc-inference-sizing/issues/31)) ([03617ce](https://github.com/gabrielkpo/hpc-inference-sizing/commit/03617ce99e67757a86c593235374bf4e9c6fbbaf))
* record the campaign's deviations from the frozen protocol and refresh the README ([#25](https://github.com/gabrielkpo/hpc-inference-sizing/issues/25)) ([0a375bc](https://github.com/gabrielkpo/hpc-inference-sizing/commit/0a375bcdfb89facc0d52bb1d007e4fcf6b786a79))

## [0.2.0](https://github.com/gabrielkpo/hpc-inference-sizing/compare/v0.1.0...v0.2.0) (2026-11-11)


### Features

* campaign runner with resume and stop rules, local SLURM cluster and job arrays (M3) ([#10](https://github.com/gabrielkpo/hpc-inference-sizing/issues/10)) ([d7d3244](https://github.com/gabrielkpo/hpc-inference-sizing/commit/d7d32449471128eb1189804af089c91956e2aecc))
* measurement bench with load sweep, streaming client and run files (M2) ([#9](https://github.com/gabrielkpo/hpc-inference-sizing/issues/9)) ([f0a2716](https://github.com/gabrielkpo/hpc-inference-sizing/commit/f0a27161a40f7da1ae0820b51cc7b355d70f4862))
* memory calculator for the KV-cache capacity of each split (M1) ([#7](https://github.com/gabrielkpo/hpc-inference-sizing/issues/7)) ([ad3b9a9](https://github.com/gabrielkpo/hpc-inference-sizing/commit/ad3b9a96c83b86839550b1f1e3951e6c54cbea14))
* test session matrix, node setup script and step-by-step runbook ([#13](https://github.com/gabrielkpo/hpc-inference-sizing/issues/13)) ([1df13a1](https://github.com/gabrielkpo/hpc-inference-sizing/commit/1df13a1196499f128b2fff67b4aaa312e9798537))
* Vast.ai driver with rental on confirmation, guard and in-instance backstop (M3b) ([#11](https://github.com/gabrielkpo/hpc-inference-sizing/issues/11)) ([634b72c](https://github.com/gabrielkpo/hpc-inference-sizing/commit/634b72ced78fa2eb8e8ee994c3ce6c7977879070))


### Bug Fixes

* read the Vast.ai instance list and credit from the live API paths ([#12](https://github.com/gabrielkpo/hpc-inference-sizing/issues/12)) ([6910eba](https://github.com/gabrielkpo/hpc-inference-sizing/commit/6910eba02015b0b1bf65f9cd270c0960ec8a886f))


### Documentation

* add the DP2+EP unit to the protocol and revise H2 ([#8](https://github.com/gabrielkpo/hpc-inference-sizing/issues/8)) ([83ab713](https://github.com/gabrielkpo/hpc-inference-sizing/commit/83ab7136da7953046937fb3914a0f1531cbf5c45))
* freeze protocol ([#15](https://github.com/gabrielkpo/hpc-inference-sizing/issues/15)) ([6588ae5](https://github.com/gabrielkpo/hpc-inference-sizing/commit/6588ae5f9e5fe3a30bb513369106186b359f7007))
* record what the test session checked and found ([#14](https://github.com/gabrielkpo/hpc-inference-sizing/issues/14)) ([6ea1e27](https://github.com/gabrielkpo/hpc-inference-sizing/commit/6ea1e2771b8db7618d6c1d1fb5d1044c5a668ca5))
* showcase README, measurement protocol and decision records ([#1](https://github.com/gabrielkpo/hpc-inference-sizing/issues/1)) ([9f53c80](https://github.com/gabrielkpo/hpc-inference-sizing/commit/9f53c8017edc3b11b2b82e1182f80943304c9064))

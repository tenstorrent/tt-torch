[![Tests][tests badge]][tests]
[![Codecov][codecov badge]][codecov]

<div align="center">

<h1>

 [Hardware](https://tenstorrent.com/cards/) | [Documentation](https://docs.tenstorrent.com/tt-torch/) | [Discord](https://discord.gg/tenstorrent) | [Join Us](https://job-boards.greenhouse.io/tenstorrent?gh_src=22e462047us) | [Bounty $](https://github.com/tenstorrent/tt-forge/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty)

</h1>
<picture>
  <img alt="tt-torch Logo" src="docs/public/images/tt-torch-logo.png" height="250">
</picture>

</div>
<br>

# tt-torch

tt-torch is a [PyTorch2.0](https://pytorch.org/get-started/pytorch-2.0/) and [torch-mlir](https://github.com/llvm/torch-mlir/) based front-end for [tt-mlir](https://github.com/tenstorrent/tt-mlir/).

-----
# Quick Links
- [Overview](https://docs.tenstorrent.com/tt-torch/overview.html)
- [Building](https://docs.tenstorrent.com/tt-torch/build.html)
- [Models](https://docs.tenstorrent.com/tt-torch/models/supported_models.html)

-----
# What is this Repo?

The tt-torch repository is a PyTorch-based front end compiler that lets developers write standard PyTorch models as well as compile and run those models on Tenstorrent AI accelerators. tt-torch is a bridge between PyTorch models, MLIR dialects (Tenstorrent-specific IRs like ttir and ttgir), and low-level hardware execution on Tenstorrent chips.

-----
# Related Tenstorrent Projects
- [tt-forge-fe](https://github.com/tenstorrent/tt-forge-fe)
- [tt-xla](https://github.com/tenstorrent/tt-xla)
- [tt-forge](https://github.com/tenstorrent/tt-forge)
- [tt-mlir](https://github.com/tenstorrent/tt-mlir)
- [tt-metalium](https://github.com/tenstorrent/tt-metal)
- [tt-tvm](https://github.com/tenstorrent/tt-tvm)

-----
# Tenstorrent Bounty Program Terms and Conditions
This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-forge, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!

[codecov]: https://codecov.io/gh/tenstorrent/tt-torch
[tests]: https://github.com/tenstorrent/tt-torch/actions/workflows/on-push.yml?query=branch%3Amain
[codecov badge]: https://codecov.io/gh/tenstorrent/tt-torch/graph/badge.svg?token=XQJ3JVKIRI
[tests badge]: https://github.com/tenstorrent/tt-torch/actions/workflows/on-push.yml/badge.svg?query=branch%3Amain

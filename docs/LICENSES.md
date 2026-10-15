# Licenses and third-party materials

| Material | License and provenance |
|---|---|
| Project-owned code | [Apache-2.0](../LICENSE) |
| Author-owned experimental data, artifacts, reports, and manuscript figures | No separate public license has been specified; code licensing does not automatically license these materials |
| WorkBuddy Bench inputs and upstream code | Tencent's custom WorkBuddy Bench license, including territorial restrictions; retain the upstream license rather than relabeling as Apache or CC-BY |
| OpenCode 1.18.16 executable and plugins | [MIT text](third_party/OpenCode_LICENSE.txt), obtained from [official v1.18.16](https://raw.githubusercontent.com/anomalyco/opencode/v1.18.16/LICENSE) on 2026-10-06 |
| Claw-Eval upstream | MIT according to the upstream README and license; retain the license accompanying the restored benchmark |
| Attachments, retrieved pages, quoted documents, and embedded software | Their respective source licenses and rights remain in effect |

Official sources: [WorkBuddy Bench](https://github.com/Tencent/workbuddy-bench), [Claw-Eval](https://github.com/claw-eval/claw-eval), and the [Claw-Eval dataset](https://huggingface.co/datasets/claw-eval/Claw-Eval).

The locally frozen Claw copy originally lacked a LICENSE and a verifiable upstream commit. An MIT license was subsequently obtained from upstream with its provenance recorded. Actual experiment file identity is established by per-file SHA-256; the upstream main branch at license retrieval is not asserted to be the executed version.

Benchmark assets are external to this package. Frozen image and binary identifiers remain part of run configuration; users supply the corresponding environment and model access. README illustrations are copied from the companion manuscript; see [figure provenance](figures/README.md).

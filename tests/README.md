# 测试

```sh
python -m pip install -e '.[analysis,runtime,dev]'
python -m pytest -q tests/unit
python -m pytest -q tests/integration/test_claw_opencode_campaign.py
python -m release.verify --data-root data
```

需要真实 Docker/OpenCode 的测试显式要求 `CLAW_EVAL_ROOT`、`CLAW_OPENCODE_BINARY` 和 `CLAW_ACTOR_IMAGE`；没有这些环境时跳过。
历史实现的测试与对应实现一并归档。公开数据校验另行检查全部文件、原生评分绑定和报告链接。

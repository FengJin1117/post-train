**工程文件夹结构**
建议把实验项目放在 `ms-swift` 仓库之外，与源码并列：

```text
workspace/
├── ms-swift/                 # 上游框架源码
└── math-reasoning/ # 你的实验工程
    ├── configs/
    ├── scripts/
    ├── datasets/
    ├── eval/
    ├── results/
    └── README.md
```

Codex 以 `workspace/` 为工作目录时，可以同时读取两个目录。安装框架时使用：

```bash
pip install -e ../ms-swift
```

这样更适合长期研究：实验配置、结果与框架源码解耦；升级 `ms-swift` 更容易；也不会把模型权重、日志和研究文档混进框架仓库。

仅当你确实要修改框架能力，例如新增 reward function、修复 Trainer 或贡献 PR 时，才在 `ms-swift` 内修改代码。自定义 GRPO reward 优先放在实验项目中，通过 `--external_plugins` 引入。

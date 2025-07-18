# CLI 工具脚本

本目录包含 Florence Forge 的命令行界面工具和相关脚本。

## 脚本列表

### florence_cli.py
主要的命令行界面工具，提供完整的 CLI 功能。

**功能特性：**
- 训练模型管理
- 数据集处理
- 配置文件管理
- 模型评估和推理

**使用示例：**
```bash
# 使用配置文件训练模型
python florence_cli.py train --config config.yaml

# 运行模型推理
python florence_cli.py inference --model path/to/model --input image.jpg

# 数据转换
python florence_cli.py convert --format coco --input data/ --output converted/
```

## 快速开始

1. **查看帮助信息：**
   ```bash
   python florence_cli.py --help
   ```

2. **查看特定命令帮助：**
   ```bash
   python florence_cli.py train --help
   ```

## 相关文档

- [CLI 用户指南](../../docs/cli/CLI_USER_GUIDE.md)
- [CLI 命令参考](../../docs/cli/CLI_COMMAND_REFERENCE.md)
- [CLI 快速参考](../../docs/cli/CLI_QUICK_REFERENCE.md)
- [CLI 故障排除](../../docs/cli/CLI_TROUBLESHOOTING.md)

## 注意事项

- 确保已安装所有必要的依赖
- 运行前请检查配置文件的正确性
- 建议在虚拟环境中运行

## 支持

如果遇到问题，请参考：
- [故障排除指南](../../docs/cli/CLI_TROUBLESHOOTING.md)
- [开发文档](../../docs/development/)
- 项目 Issues 页面
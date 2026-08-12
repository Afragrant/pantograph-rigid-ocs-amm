# Pantograph-Rigid OCS via Assumed Modes Method

基于假设模态法（Assumed Modes Method）的受电弓–刚性接触网（Rigid Overhead Contact System）动力学仿真。

## 特性

- 受电弓模型：DSA380 / DSA250 / TSG18F（`procs.py` 中 `PANTOGRAPH` 切换）
- 刚性接触网：3 组参数预设（`RIGID_OVERHEAD_CONTACT_SYSTEM` 切换）
- 接触求解：Newton / 有效集（active-set）迭代，Woodbury 公式加速
- 接触力后处理：6 阶 Butterworth 低通滤波（20 Hz）
- 输出：接触力 / 弓头位移 / 接触网位移（全程 + 稳定段）

## 环境

使用 [pixi](https://pixi.sh) 管理依赖（Python 3.13）：

```bash
pixi install
```

依赖见 `pixi.toml`（numpy、scipy、matplotlib、pandas、openpyxl、tqdm）。

## 使用

```bash
pixi run python procs.py  # 运行仿真
```

- 仿真结果与图输出到 `result/`（gitignore）

## 参考文献

1. 关金发. 受电弓与刚性接触网动力相互作用研究[D]. 西南交通大学, 2018.
2. 陈龙. 高速刚性接触网-受电弓系统仿真及动态性能研究[D]. 西南交通大学, 2024.

## License

MIT

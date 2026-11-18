# 实现审计：`crd_backbone.py` / `crd_blip.py` / `crd_offset.py`

> 2026-11-09。行号对应当前工作副本。逐条回答五个问题，附代码位置与公式。

---

## Q1. 标准化的公式、统计量在哪个 split 上计算

### `crd_backbone.py` L242–L246

```python
242:    Str = bb.score(Utr, Vtr)
243:    # standardise the native score with TRAIN statistics so CRD is scale-free
244:    mu, sd = float(Str.mean()), float(Str.std() + 1e-8)
245:    Str, Sva, Ste = (Str - mu) / sd, (Sva - mu) / sd, (Ste - mu) / sd
```

公式：

```
μ = mean( s_train )                  # 逐行平均，行 = 一个 (I,T) 对
σ = std( s_train, ddof=0 ) + 1e-8    # 总体标准差 + 数值保护
z = (s − μ) / σ
```

**统计量只来自 train**（`Str` 来自 `load("train")`），`Sva` / `Ste` 用同一条 `(μ, σ)` 变换。

### `crd_blip.py` L207–L209

```python
207:    mu, sd = float(Str.mean()), float(Str.std() + 1e-8)
208:    Str, Sva, Ste = (Str - mu) / sd, (Sva - mu) / sd, (Ste - mu) / sd
```

完全相同的公式与 split。

### `crd_offset.py`

**不做标准化**。L127–L134 直接以原始余弦为目标：

```python
127:    def cycle_means(d, U, V):
128:        s = (U * V).sum(1)
129:        m = pd.Series(s).groupby(d.cycle_id.values).transform("mean").to_numpy()
132:    _, mtr = cycle_means(tr, Utr, Vtr)
```

---

## Q2. 统计量是否只用 train、是否固定套用到 val/test

**是。** `crd_backbone.py` L245 / `crd_blip.py` L208 都只计算一次 `(μ, σ)`（来自 train），
val 与 test 用同一条变换，**没有任何按 split 重新计算**的路径。

代码里不存在 `Sva.mean()` 或 `Ste.mean()` —— 可用如下命令核验：

```bash
rg "Sva\.(mean|std)|Ste\.(mean|std)" fgbench/models/
# 无输出
```

**无泄漏**：val/test 的分数从未参与 μ、σ 的计算。

---

## Q3. 回归目标、预测偏移与最终分数是否同一尺度

三个量全部在 **z 空间**：

| 量 | 定义 | 代码 |
|---|---|---|
| 回归目标 | `m = z`（row）或 `m = mean_c(z)`（cycle） | `crd_backbone` L247–L249；`crd_blip` L214–L216 |
| 头部输出 | `m̂ = a(u) + b(v) + b0`，拟合 `z` | L260 / L224，`F.huber_loss(head(u,v), m)` |
| 最终分数 | `z' = z − λ·m̂` | L279 / L241 |

### 尺度还原：不需要显式还原，但关系必须写清

```
z 空间：  z' = z − λ·m̂
原始空间：s' = μ + σ·z' = s − λ·σ·m̂
```

- **对排序无影响**：AUROC / AUPRC 对正仿射变换不变，所以报告 z 空间的 AUC 等于原始空间的 AUC。
- **但 λ 的物理含义变了**：`crd_backbone` 的 λ 是 **z 单位**下的系数，
  等效的原始空间系数是 **λ·σ**。这就是同一个 Attribute 在两套管线里选出 0.7 与 0.8 的原因，
  也是两个 λ 不能直接比较的原因。
- 头部文件里保存了 `mu, sd`（L304–L306），因此 `s' = s − λσ·m̂` 可精确还原。

`crd_offset.py` 的目标与推理都在原始空间（L232: `crd=cos_te - lam*mh_te`），自身一致，
但与 `crd_backbone` 的 λ 不可比。

---

## Q4. λ 由谁选、是否有 test 信息参与

### `crd_backbone.py` L274–L277

```python
274:    yv, yt = va.pair_label.to_numpy(), te.pair_label.to_numpy()
275:    lam = args.lam_fixed if args.lam_fixed >= 0 else max(
276:        np.round(np.arange(0, 2.01, 0.1), 1),
277:        key=lambda L: roc_auc_score(yv, -(Sva - L * mh_va)))
```

- 网格 `0.0 → 2.0`，步长 `0.1`（含 λ=0，即"不校正"）。
- 目标函数只用 **val 的标签 `yv`、val 的分数 `Sva` 与 val 的预测偏移 `mh_va`**。
- **test 从未参与**：`mh_te`、`Ste`、`yt` 在 L279 之后才第一次进入计算。
- 并列时取较小 λ（`max` 返回第一个最大值；`crd_offset` L213 用 `a > best[1]` 亦是同一行为）。

`crd_blip.py` L236–L239 与 `crd_offset.py` L207–L215 逻辑相同。

**结论：无 test 泄漏。**

---

## Q5. 三种子是否使用相同的标准化统计量协议

**协议相同，数值在实际运行中也相同，但存在一个隐性依赖。**

```python
193:    def load(split, limit_cycles=0):
195:        tag = f"{args.backbone}_{args.dataset}_{args.facet}"
196:        if limit_cycles:
197:            tag += f"_c{limit_cycles}"
198:        cache = OUT / f"crdbb_{tag}_{split}.npz"
199:        if cache.is_file():
201:            return pd.read_parquet(...), z["U"], z["V"], z["S"]
210:        if limit_cycles and d.cycle_id.nunique() > limit_cycles:
211:            cids = d.cycle_id.unique(); rng.shuffle(cids)   # ← rng 由 args.seed 播种
212:            d = d[d.cycle_id.isin(cids[:limit_cycles])]
```

- train 子集抽样用 `rng = default_rng(args.seed)`，**不同 seed 会抽到不同子集**；
- 但缓存名**不含 seed**（只有 `_c{cycles}`），所以第一次运行（通常是 seed 0）建好缓存后，
  seed 1 / 2 直接复用同一份 `U, V, S`（L199–L201）；
- 于是 **三个种子的 μ、σ 逐位相同**，标准化协议一致。

**风险**：若删除缓存并用 seed=1 重建，train 子集与 μ/σ 都会变，结果不可复现。
→ 必须在数据卡中固定"子集由首个运行 seed 决定"这一事实，或把 seed 写进缓存名（推荐后者）。

---

## 审计发现（按严重程度）

| # | 问题 | 影响 | 建议 |
|---|---|---|---|
| 1 | **`crd_blip.py` L265 保存的头部不含 `lam`** | 外部评测脚本用 `ck.get("lam", 1.0)` 回退到 1.0，而 val 选出的 λ 是 0.8–1.4（Attribute ITC 0.9 / ITM 0.8）→ SugarCrepe / SeeTRUE / What'sUp 的 BLIP 结果全部用错 λ | 保存时加入 `lam`；外部结果若再用需重跑 |
| 2 | `crd_offset.py` L239 只存 `state_dict()` | 无 `mu/sd/lam`，还原须查 json/npz | 旧结果保留即可，不再新增 |
| 3 | 缓存名不含 seed（L195–L197） | 重建缓存可改变 train 子集与 μ/σ | 缓存名加入 seed，或在数据卡锁定 |
| 4 | λ 的尺度 | z 单位 λ 与原始空间 λ 不可比（等效系数 λ·σ） | 论文中明确写下 `s' = s − λσ·m̂` |

---

## 结论

**三项前提全部满足**：

1. **无数据泄漏** —— μ、σ 只来自 train；λ 只来自 val；test 只在 λ 固定后进入计算。
2. **训练目标与推理公式尺度一致** —— 目标、偏移、最终分数同在 z 空间，
   且 `s' = s − λσ·m̂` 可由保存的 `mu/sd` 精确还原。
3. **多种子协议一致** —— μ/σ 在三种子间逐位相同（依赖缓存复用，见发现 #3）。

**因此可以统一采用 `crd_backbone.py` 作为正式方法实现**，
`crd_offset.py` 的结果保留为历史口径，不进入最终主表。

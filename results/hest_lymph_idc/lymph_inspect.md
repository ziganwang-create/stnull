# LYMPH_IDC (HEST-bench) 摸底 + 协议决定

日期 2026-08-24。数据只读:`d:\project\STImage\Xiao_data_codes\data\hest_bench\LYMPH_IDC\`
产出:本文件 + `lymph_depth_pretable.csv`(36 列,8 行 = 4 样本 × 2 spot 集)+ `lymph_caliber_moran.csv`。
脚本(scratchpad,可重跑):`lymph_inspect.py` / `lymph_pass2.py` / `pass3.py`。

**本文件中所有零模型强度数字都是「预注册预期」,不是实测。审计尚未跑,刻意没跑。**

---

## 1. 结构

| 项 | 值 | 来源 |
|---|---|---|
| 样本数 | **4**(NCBI681/682/683/684) | `adata/*.h5ad` |
| 基因数 | **33931**,四样本 `var/_index` 逐字节相同 | `VAR_IDENTICAL_ACROSS_SAMPLES: True` |
| spot 数 | 4992 / 4988 / 4992 / 4992(= 完整 Visium 78×128 网格) | `obs/_index` |
| X | **原始整数计数**,CSR float32,最小值 1,最大值 2767–10975;`obs/total_counts` 与行和一致 | `is_integer=True` |
| 坐标 | `obsm/spatial` = (pxl_col, pxl_row) = (x, y);`obs/pxl_row_in_fullres` / `pxl_col_in_fullres`,单位 **全分辨率像素** | 逐列比对确认 |
| spot 直径 | 200.8–201.2 px(full-res);patch 409 px 下采样到 224 | `uns/spatial/ST/scalefactors`, `patches` attrs |
| 患者字段 | **h5ad 里没有**。见 §2 的推断 | `obs` 只有 QC 列 |
| 条码 | 无重复(682 是 4988 个 spot / 4988 个唯一条码,不是重复) | `n_unique_barcodes` |
| patches | `patches/*.h5`:`img (N,224,224,3)`、`barcode`、`coords`,**N 与 adata 完全相同且同序** | `same_order=True` |

## 2. splits 粒度与患者结构

`splits/train_{0..3}.csv` + `test_{0..3}.csv` = **留一样本**,每折 3 训 / 1 测。

样本↔患者映射:数据目录里没有 patient 列,但由建集代码可反推 ——
`HESTData.py:1247` `create_benchmark_data` 里
`splits = meta_df.groupby(['dataset_title','patient'])['id'].agg(list)`,
再 `create_splits(..., K)`(`HESTData.py:1273`)。若 K ≠ 分组数会走重分配分支,产生 size ≥ 2 的测试集;
这里 4 折每折测试集恰好 1 个样本 → 分组数必须 = 4 → **4 个样本 = 4 个患者,一人一片**。

**结论:样本级即患者级,官方 splits 无泄漏。** 这一点与 TRIPLEX 那条审计发现相反(那里是
leave-one-patient-out 被标成 slide-level);LYMPH_IDC 这里两者恰好重合。协议上我们只需在
报告里显式声明该重合,不需要另建划分。

⚠ n=4 的代价:每折训练侧只有 3 个患者,跨切片统计量(如 corr(sd(log lib), r))只有 4 个点,
**无法在本队列内部重估 H3 的 0.964**;LYMPH 只能作为 HER2ST 拟合关系的外部验证点。

## 3. 最重要的发现:官方 spot 集包含大量组织外背景 spot

`obs/in_tissue` 同时含 0 和 1,而 h5ad 与 patches 都保留了**全部** 4992 个网格 spot;
`benchmark.py:288` 用 patch 文件的 barcodes 去 subset adata → **HEST-bench 官方评测确实把组织外
spot 一起算进去了**。

| 样本 | in_tissue=1 | in_tissue=0 | 组织外占比 | 组织外 lib 中位 | 组织内 lib 中位 |
|---|---|---|---|---|---|
| NCBI681 | 4553 | 439 | 8.8% | 2159 | 25809 |
| NCBI682 | 4859 | 129 | 2.6% | 20 | 24119 |
| NCBI683 | 4454 | 538 | 10.8% | 2053 | 18997 |
| NCBI684 | 3055 | **1937** | **38.8%** | 1649 | 22140 |

这些 spot 的图块就是白片:随机抽 120 个测像素统计 ——

| | patch 均值 | patch 标准差 |
|---|---|---|
| in_tissue=0 | 241.2 / 241.3 | 20.5 / 17.3 |
| in_tissue=1 | 203.2 / 175.8 | 39.2 / 50.1 |

**后果(这是本队列与 HER2ST 的关键差异)**:组织/背景边界同时被图像(一个亮度阈值就够)
和文库零模型(深度差 10 倍)看见。HER2ST 上 r_lib 36/36 完胜图像岭回归,是因为图像看不到深度;
**LYMPH_IDC 的 all_spots 口径下,图像能看到同一个混杂轴**,所以两边会一起被抬高,
岭回归很可能追平甚至反超文库零模型 —— 那不是图像学会了生物学,是两者都在读同一条组织边界。

## 4. 深度结构预表(`lymph_depth_pretable.csv`)

HER2ST 参照(`lib_structure_audit.csv`,36 切片):sd(log lib) 中位 **0.696**(0.389–1.428);
Moran's I(log lib) 中位 0.618;实测 libonly r 中位 0.230。

| 样本 | spot 集 | n | sd(log lib) 全转录组 | Moran I(log lib) | sd×Moran |
|---|---|---|---|---|---|
| NCBI681 | all | 4992 | 1.227 | 0.807 | 0.990 |
| NCBI681 | in_tissue | 4553 | **0.608** | 0.757 | 0.461 |
| NCBI682 | all | 4988 | 1.629 | 0.903 | 1.471 |
| NCBI682 | in_tissue | 4859 | **1.290** | 0.923 | 1.191 |
| NCBI683 | all | 4992 | 1.169 | 0.891 | 1.041 |
| NCBI683 | in_tissue | 4454 | **0.742** | 0.819 | 0.608 |
| NCBI684 | all | 4992 | **2.265** | 0.952 | 2.156 |
| NCBI684 | in_tissue | 3055 | **0.801** | 0.681 | 0.545 |

稀疏度(官方 hest_var50 面板,零计数占比 / ≤1 计数占比):

| 样本 | all_spots | in_tissue | 面板行和中位 | 面板行和 =0 的 spot 数(all / in_tissue) |
|---|---|---|---|---|
| NCBI681 | 0.619 / 0.825 | 0.588 / 0.810 | 46.5 | 118 / 12 |
| NCBI682 | 0.446 / 0.621 | 0.432 / 0.611 | 107 | 214 / 119 |
| NCBI683 | 0.655 / 0.838 | 0.625 / 0.822 | 49 | 101 / 31 |
| NCBI684 | 0.717 / 0.880 | 0.594 / 0.819 | 27 | 738 / 25 |

对照 HER2ST:面板 0/1 计数占 59%。**LYMPH 官方 50 基因面板的 ≤1 占比 61–88%,比 HER2ST 更稀疏**,
面板行和中位只有 27–107 个计数(HER2ST 面板行和量级远高于此)。H2 的机制(log1p-CP10K
在稀疏计数上的代数伪影)在这里应当**更强**,如果我们跑 CP10K 口径的话。

表达自相关远低于深度自相关(`lymph_caliber_moran.csv`,50 基因中位 Moran I,log1p-raw 口径):
0.146 / 0.415 / 0.130 / 0.250(all_spots),而同样 spot 集的 log lib Moran I 是 0.807–0.952。
**深度场比表达场空间结构强得多** —— 与 H3「难易 ≈ 深度方差地形图」一致。

单基因层面的深度耦合:median_j |corr(log1p(count_j), log lib)| = 0.264 / 0.380 / 0.232 / 0.369
(all_spots)。HEST 官方口径下目标**逐基因**直接与深度耦合。

---

## 5. 预注册预期(在跑任何零模型之前写下)

用 HER2ST 36 切片拟合 libonly_median_r ~ sd(log lib):
`r = 0.0296 + 0.2874 × sd(log lib)`,corr = **0.9637**,sd 覆盖 0.389–1.428。
(备用预测子 sd×Moran:`r = 0.1163 + 0.2590 × x`,corr = 0.9337。)

**P1(主预测,all_spots = HEST 官方口径)** 文库零模型逐切片中位 r:

| 样本 | 预测 r(sd 版) | 预测 r(sd×Moran 版) | 外推? |
|---|---|---|---|
| NCBI683 | 0.366 | 0.386 | 否 |
| NCBI681 | 0.382 | 0.373 | 否 |
| NCBI682 | 0.498 | 0.497 | **是**(超出 HER2ST sd 上界) |
| NCBI684 | 0.681 | 0.675 | **是**(sd=2.27,远超 1.428) |

→ 四切片中位预测 **约 0.44 ± 0.05**,即 **明显强于 HER2ST 的 0.230**(约 1.9 倍)。
排序预测(可证伪、且与"外推"无关):**NCBI684 > NCBI682 > NCBI681 ≳ NCBI683**。
两个外推点按外推处理,只当作"会很高",不当作点估计。

**P2(in_tissue 限制口径)** 同一公式:0.204 / 0.400 / 0.243 / 0.260 → 中位 **约 0.25**,
即 **回落到 HER2ST 的 0.230 水平**。

P1 与 P2 的差(0.44 vs 0.25)几乎全部来自组织外背景 spot,**不来自组织学**。
这是本次跨队列检验最锋利的一刀:如果实测复现 P1/P2 这一对,那么
"HEST-bench 上文库零模型很强"这件事就被定位成**评测集构造(未过滤背景 spot)** 的产物,
而不是 LYMPH 这个器官/疾病的性质。

**P3(口径对比 = H2 的检验)** HEST 官方口径是 `log1p(raw counts)`,**没有任何 CP 归一化**
(见 §6 证据)。目标里直接保留文库大小 → 文库零模型在官方口径下应当**显著高于**同 spot 集的
CP10K 口径。预期 log1p-raw 口径的 r 比 CP10K 口径高 **约 0.10–0.20**;若不然,
说明深度信号已被面板稀疏性吃掉。

**P4(零生物学模拟)** 鉴于面板 ≤1 计数占比 61–88% > HER2ST 的 59%,
CP10K 口径下零生物学模拟应当 **≥ 真实数据**(HER2ST 是 0.266 ≥ 0.230,34/36)。
预期 LYMPH 上 4/4 切片满足,且差距更大。

**P5(H5 空间羞辱线)** 表达 Moran I 只有 0.10–0.42,低于 HER2ST 常见水平,
所以 r_nbr(4/6 邻居真值均值)在 LYMPH 上**相对**弱一些;但图像岭回归的绝对水平同样低,
因此仍预期 r_nbr 在 ≥ 3/4 切片上打赢图像岭回归。这条我信心最低,明确标为弱预测。

**证伪条件**:若 all_spots 口径下实测中位 r 落在 0.20–0.28(即 ≈ HER2ST 且 ≈ P2),
则 H3 的线性外推在本队列失败,"混杂量级由 sd(log lib) 驱动"需要加饱和项;
若排序与 P1 的排序不一致(尤其 NCBI684 不是最高),H3 的核心主张就被证否。

---

## 6. 协议决定

**(a) 划分**:直接用官方 `splits/*.csv`(留一样本),并在报告里写明
「本队列样本级 = 患者级(4 患者 × 1 片,依 `HESTData.py:1273` 的分组逻辑反推)」。
不另建划分。n=4 折,所有跨折统计只报中位与逐折值,不报 p 值。

**(b) spot 集:必须跑两套,并列报告。**
- `all_spots` = HEST 官方实际评测的 spot 集(含 439–1937 个组织外 spot)——用于"复现官方数字所在的轴";
- `in_tissue` = `obs['in_tissue']==1` ——用于"去掉组织/背景混杂后还剩什么"。
两者之差本身就是本次审计最主要的产出之一,不是稳健性附录。

**(c) 基因面板:报三档,主结论用训练侧面板。**
1. `hest_var50`(官方 `var_50genes.json`)—— **必须报**,因为这是官方数字所在的面板;
   同时必须标注它是**跨含测试样本的全部 4 个样本 + 全部 4992 个 spot(含背景)选出来的**(§6 证据)。
2. `trainHVG50` —— 每折只用 3 个训练样本重选 50 个 HVG(同一算法)。已算好,存于
   scratchpad `lymph_panels.json`。**与官方面板重叠仅 16–24/50** → 官方面板里有 26–34 个基因
   在训练侧选不出来,这就是选择泄漏的量化。
3. `trainHVG737` —— 与 HER2ST 的 737 档对齐,便于跨队列比较面板大小效应。已算好。

**(d) 两个口径(target space)**
- `hest_log1p_raw`:`log1p(count)`,**无 CP** —— 官方口径。
- `log1p_cp10k_panel`:`log1p(count / panel_row_sum × 1e4)` —— 我们 HER2ST 侧的口径。
  ⚠ 面板行和为 0 的 spot 会产生 0/0。all_spots 下有 101–738 个这样的 spot,
  in_tissue 下仍有 12–119 个。**必须先定规则**:建议这些 spot 的该口径目标置为 0
  (等价于 log1p(0)),并在报告里给出受影响 spot 数;不要直接 dropna(会让两个口径的 spot 集不同,
  破坏可比性)。
- 建议再加一档 `log1p_cp10k_fulllib`(用全转录组 lib 而非面板行和归一),因为它把"面板选择"
  与"深度归一"两个因素解耦。lib 偏移用**全转录组** `obs/total_counts`,不用面板行和
  (stnull 的 `LibSizeLooksLikePanelSum` 检查就是防这个)。

**(e) lib_size 定义**:一律用 `obs/total_counts`(= 全 33931 基因行和,已验证与 X 行和一致)。

---

## 7. 数据坑清单

1. **组织外 spot 未过滤**(§3)。最大的坑。NCBI684 有 38.8% 的 spot 是白片。
2. **`obsm/spatial` 与 `patches/coords` 的轴顺序相反**:
   `obsm/spatial` = (pxl_col, pxl_row) = (x, y);`patches/coords` = (pxl_row, pxl_col) = (y, x)。
   逐样本比对确认。做 kNN/Moran 不受影响(只是转置),但**任何把 patch 贴回图像坐标的代码会错**。
3. **NCBI684 有 1 个 spot 全转录组 total_counts == 0**。log(lib)、任何 CP 归一化都会炸。
   in_tissue 子集里没有这个 spot(min lib in_tissue = 48)。另外 all_spots 下 lib<100 的 spot 有
   78 / 230 / 69 / 630 个。
4. **面板行和为 0** 的 spot(见 6d):CP10K-panel 口径下 0/0。
5. **anndata / scanpy 均未安装**。h5ad 用 h5py 直读即可(X 是 `csr_matrix` 编码,
   `X.attrs['shape']` 给形状;字符串列是 object dtype 需 `.decode()`;
   `var/feature_types`、`var/genome` 是 categorical 编码,取 `categories[codes]`)。
6. **`obs/pct_counts_mito` 在 NCBI684 上中位为 NaN**(线粒体基因 13 个,多数 spot 全 0)。别用它做 QC 过滤。
7. **`X` 是原始计数,不是已归一化的**。但 HEST-bench 的 pipeline 会在读取时 log1p —— 注意别 log 两次。
8. 每个 h5ad 里都塞了 `uns/spatial/ST/images/downscaled_fullres`(约 1000×990×3 uint8),
   所以文件有 190–256 MB;只要表达就别整文件 load。

## 8. 顺带确认/纠正的两条 HEST 源码审计结论

**(i) `normalize_adata` 的 docstring 与代码不符 —— 确认。**
`methods\gen4_hest\src\hest\bench\st_dataset.py:55-79`:docstring 写
"Normalize each spot by total gene counts + Logarithmize each spot",
函数体只有 `sc.pp.log1p(filtered_adata)`,**没有任何 normalize_total**。
`benchmark.py:96` 的 config `normalize: bool = True` 走的正是这个函数。
(注意 `hest/utils.py:396` 另有一个同名函数确实做 `normalize_total(target_sum=1)` 再 `×1e6`,
但 bench 路径**不**调它 —— 引用时必须写清是哪一个,否则会被反驳。)
另:`load_adata`(`st_dataset.py:81`)先 `adata[:, genes]` 切到 50 基因**再**归一化;
因为归一化实际只有 log1p,顺序在此无影响 —— 但若有人"修好"了 CP 归一化,它会变成按面板行和归一。

**(ii) 50 基因面板跨含测试样本的全部样本选一次 —— 确认,且已逐基因精确复现。**
`HESTData.py:1247` `get_k_genes_from_df(meta_df, 50, 'var', ...)` → `utils.py:671 get_k_genes`:
先按 `min_cells_pct=0.10` 逐样本过滤取交集(本数据集 4 样本交集 = **10276** 个基因),
再把所有样本的 spot **拼在一起**,`sc.pp.log1p` + `sc.pp.highly_variable_genes(n_top_genes=50)`。
我用 numpy 重实现 scanpy 的 `flavor='seurat'`(expm1 回计数 → dispersion=var/mean →
log-dispersion 在 20 个 log1p(mean) 等宽箱内 z 分)得到 **50/50 完全一致**的基因集。
这同时证明:panel 是在 **4 个样本 × 全部 4992 个 spot(含组织外背景)** 上选的。
量化影响:改成每折只用 3 个训练样本重选,与官方面板重叠只有 **16–24 / 50**。

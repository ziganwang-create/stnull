# LYMPH_IDC 零像素审计(跨队列检验 HER2ST 结论)

日期:2026-08-24。数据:`Xiao_data_codes\data\hest_bench\LYMPH_IDC\`(HEST-bench,4 样本 = 4 患者,
淋巴结转移性 IDC,Visium 全网格 4988–4992 spot/样本,X = 原始整数计数,已验证)。
划分:官方 `splits/*.csv` 留一样本(样本级 = 患者级,见 `lymph_inspect.md`)。
脚本:scratchpad `lymph_audit_main.py`、`lymph_stnull_dogfood.py`。
数字来源:`lymph_audit_{libnull,zerobio,structure,ceiling}.csv`、`lymph_audit_structure_corr.csv`、
`lymph_stnull_dogfood.csv`、`lymph_stnull_dogfood_notes.txt`。
n=4 折,只报中位与逐样本值,不报 p 值;跨样本聚合数字读作「约 ±0.02–0.05」。

## 0. 先说结论(与 HER2ST 并排)

| 量 | HER2ST(36 切片,737 HVG,cp10k:panel) | LYMPH_IDC 最可比条件(trainHVG737,cp10k:panel) | 判定 |
|---|---|---|---|
| H1 文库零模型中位 r | **0.230** | all_spots **0.270** / in_tissue **0.209** | **复现**(量级相同或更高) |
| 图像岭回归参照 | 0.113(36/36 输给零模型) | HEST 官方排行(mean-r)0.2305–0.2842 vs 零模型官方口径 mean-r **0.3263** | **复现**:零模型赢全部 26 个已发表模型 |
| H2 零生物学 sim vs real | 0.266 ≥ 0.230(34/36) | 737 面板 16/16 行 sim≥real;官方 50 面板 12/16 | **复现**(见 §2,唯一例外 = 测序最深的 NCBI682) |
| H3 corr(sd(log lib), r) | 0.964 | n=4:0.86–0.99(cp10k);排序预测 4/4 命中(all_spots,cp10k 条件;官方口径见 §3 披露) | **部分复现**:排序/单调性成立,**线性外推在 sd>1 处证伪,需饱和项** |
| H4 r_tech(修正估计量) | 0.51(0.36–0.65) | 737/cp10k:panel:all 0.522 / in_tissue 0.446;官方条件(50 面板,log1p:raw,all)0.539 | **复现**(天花板同样 ~0.5) |
| H5 r_nbr vs 图像 | 多切片打赢 | r_nbr(k=6,官方口径 mean-r)**0.4244**,4/4 样本高于排行榜最高 0.2842 | **强复现** |

**一句话**:HER2ST 的结论不是该数据集的特性,是任务/评测构造的特性。并且 LYMPH_IDC 暴露了
HEST-bench 的两个自有放大器:(a) 官方口径 log1p(raw) 让目标直接携带深度;(b) 官方 spot 集
含 2.6–38.8% 组织外背景 spot,把零模型再抬 +0.06–0.09。

## 1. H1 文库零模型(逐基因 [1, log lib] 线性,训练侧拟合,官方折)

来源:`lymph_audit_libnull.csv`(fit=trainfit;cv 列为节内 5 折交叉拟合,供 stnull 对照)。
lib = 全转录组 `obs/total_counts`。cp10k:panel 的面板行和为 0 的 spot 目标置 0(逐样本受影响
spot 数见 `lymph_inspect.md`)。

### 逐样本中位 r(med_r_trainfit)

| 样本 | 面板 | all, log1p:raw | all, cp10k:panel | all, cp10k:full | in_tissue, log1p:raw | in_tissue, cp10k:panel | in_tissue, cp10k:full |
|---|---|---|---|---|---|---|---|
| NCBI681 | hest_var50 | 0.263 | 0.253 | 0.150 | 0.244 | 0.214 | 0.098 |
| NCBI682 | hest_var50 | 0.378 | 0.356 | 0.260 | 0.379 | 0.334 | 0.204 |
| NCBI683 | hest_var50 | 0.226 | 0.206 | 0.118 | 0.192 | 0.160 | 0.085 |
| NCBI684 | hest_var50 | 0.368 | 0.363 | 0.226 | 0.278 | 0.226 | 0.093 |
| **中位** | hest_var50 | **0.315** | **0.305** | 0.188 | **0.261** | **0.220** | 0.095 |
| NCBI681 | trainHVG737 | 0.327 | 0.253 | 0.180 | 0.317 | 0.191 | 0.129 |
| NCBI682 | trainHVG737 | 0.346 | 0.287 | 0.190 | 0.329 | 0.243 | 0.120 |
| NCBI683 | trainHVG737 | 0.288 | 0.240 | 0.151 | 0.248 | 0.187 | 0.105 |
| NCBI684 | trainHVG737 | 0.434 | 0.389 | 0.269 | 0.333 | 0.227 | 0.107 |
| **中位** | trainHVG737 | **0.337** | **0.270** | 0.185 | **0.323** | **0.209** | 0.114 |

(trainHVG50 面板与 hest_var50 同量级,略低,见 CSV。)

**披露(trainHVG 面板选择的 spot 集不匹配)**:trainHVG 面板在 all_spots 上选出,而 in_tissue
各列的分析在 in_tissue spot 集上做——面板选择 spot 集与分析 spot 集不一致。该选择纯在训练侧
完成、不触及测试样本,**无测试泄漏**;但含背景 spot 选出的 HVG 更偏深度敏感基因,方向上
**有利于零模型**,幅度未测(未重选 in_tissue-only 面板对照)。

### 图像侧参照(HEST 官方排行,`methods\gen4_hest\README.md` 表,03.04.26)

官方指标 = 50 基因逐基因 Pearson 取**均值**、再对 4 折取均值(`benchmark.py:327-348`),
口径 = log1p(raw)、all_spots、hest_var50 面板。同指标同条件下:

- **零像素文库零模型 mean-r = 0.3263**(逐折 0.268 / 0.384 / 0.276 / 0.377);
- 排行榜:最好 GenBio-PathFM **0.2842**,H-Optimus-1 0.2774,…,ResNet50 0.2305(共 26 个模型);
- → **零像素零模型高于排行榜全部 26 个已发表模型约 0.04–0.10**。in_tissue 下零模型 mean-r 仍有
  0.2916(**跨 spot 集对比**:排行榜数字是 all_spots 口径,该比较只作参考,非同 spot 集)。
- 模型计数脚注:README 自述 25 个模型,但其表格实有 **26 行**;本文按表格实数引用 26。
- 注意读法:零模型读的是测试 spot 的真实测序深度(图像拿不到的量),它是**审计基线不是竞争者**
  ——含义是「LYMPH_IDC 排行榜上 0.23–0.28 的图像分数,不高于一个不看图像的深度读出」。

## 2. H2 零生物学模拟(训练侧全局谱 + 真实逐 spot 深度多项抽样)

来源:`lymph_audit_zerobio.csv`。**预先写下的口径预期**(见 `lymph_audit_main.py` 头部,算前预注册):
log1p:raw 下目标直接携带深度,预期 sim≥real 同样成立、伪影绝对量更大。

| 条件(4 样本中位) | real | sim(零生物学) | sim≥real |
|---|---|---|---|
| hest_var50, all, log1p:raw | 0.315 | **0.415** | 3/4 |
| hest_var50, all, cp10k:panel | 0.305 | **0.381** | 3/4 |
| hest_var50, in_tissue, log1p:raw | 0.261 | **0.385** | 3/4 |
| hest_var50, in_tissue, cp10k:panel | 0.220 | **0.315** | 3/4 |
| trainHVG737, all, log1p:raw | 0.337 | **0.425** | 4/4 |
| trainHVG737, all, cp10k:panel | 0.270 | **0.346** | 4/4 |
| trainHVG737, in_tissue, log1p:raw | 0.323 | **0.407** | 4/4 |
| trainHVG737, in_tissue, cp10k:panel | 0.209 | **0.266** | 4/4 |

- 合计 28/32 行 sim≥real(HER2ST:34/36)。**零生物学(全局单一表达谱)不但复现文库零模型,
  还普遍超过真实数据** → 文库零模型的强度主要是稀疏计数 + 变换的代数结果,非生物学。跨队列复现。
- 唯一例外:NCBI682 在官方 50 面板上 real>sim(4 个条件均如此)。NCBI682 恰是测序最深、面板最不
  稀疏的样本(面板零占比 0.446 vs 其余 0.62–0.72),即**真实生物学超出伪影的正是稀疏度最低的样本**
  ——这本身支持「伪影由稀疏度驱动」的机制解释。
- 口径预期兑现:log1p:raw 的 sim−real 差普遍大于 cp10k(如 hest50 in_tissue:+0.124 vs +0.095)。

## 3. H3 深度方差结构(预注册预测逐条对答案)

来源:`lymph_audit_structure.csv`、`lymph_audit_structure_corr.csv`;预测出自 Inspect 阶段
预注册(HER2ST 36 切片拟合 r = 0.0296 + 0.2874·sd(log lib),corr 0.964,拟合域 sd∈[0.39,1.43])。

| 预注册项 | 预测 | 实测 | 判定 |
|---|---|---|---|
| P1 点估计(all, cp10k)中位 ~0.44 | 0.44±0.05 | hest50 0.305 / 737 **0.270** | **未命中**;737 落入预先写明的证伪带 0.20–0.28 → **线性外推证伪,需饱和项** |
| P1 排序 684>682>681>683 | — | all/cp10k 两个面板均**完全一致**;条件覆盖见表下披露 | **命中 4/4(cp10k 条件)** |
| P2(in_tissue)中位 ~0.25 | 0.204/0.400/0.243/0.260,中位 0.25 | 观测 0.209(737)/0.220(hest50);逐样本 sd≤0.81 的三点残差 −0.014~−0.056,sd=1.30 的 NCBI682 残差 −0.160 | **域内命中,域外过冲**(同一饱和结论) |
| P3 raw 比 cp10k 高 0.10–0.20 | +0.10~0.20 | 方向 12/12 行为正,但幅度 +0.01~+0.11(737 in_tissue 最大 0.114) | **方向命中,幅度低于预测**——深度信号部分被面板稀疏性吃掉(预测里预留的备择解释) |
| P4 sim≥real 4/4 且差距更大 | 4/4 | 见 §2:737 面板 4/4,官方面板 3/4;sim−real 差(约 +0.06~+0.12)大于 HER2ST 的 +0.036 | **基本命中** |
| P5 r_nbr ≥3/4 切片打赢图像 | ≥3/4 | 4/4 样本 r_nbr(mean-r 0.331–0.553)> 排行榜最高 0.2842 | **命中** |
| corr(sd(log lib), r) | HER2ST 0.964 | n=4:all/cp10k hest50 0.860、737 **0.992**;in_tissue 0.85–0.88(737/raw 例外 0.35) | 与 HER2ST 一致(n=4 只作外部验证点) |

**披露(排序命中的条件覆盖)**:上表「命中 4/4」的排序命中成立于 **cp10k 条件(两个面板)**,
以及 737/log1p:raw 的 3/4(682 与 684 换位)。而在**官方口径组合(hest_var50 + log1p:raw)下,
观测排序为 682>684**(0.378 vs 0.368,差 0.011,在本报告 ±0.02–0.05 的噪声带内)——预注册中
「排序不符即证伪」的条款在该条件下被**字面触发**。辩护理由一并写明:HER2ST 的拟合关系
r = 0.0296 + 0.2874·sd 基于 cp10k 口径建立,其预测适用域是 cp10k;log1p:raw 口径的目标额外
直接携带深度,不在原拟合的口径域内。但该辩护是事后限定,预注册文本未预先区分口径域,如实记录。

**披露(sd(log lib) 定义差)**:depth 预表(pretable)阶段用的是 log(lib+1),audit 主脚本按
核查后统一定义用 log(max(lib,1)),两者在本数据上 sd 差约 0.01–0.016(不改变任何排序)。后续
所有队列已统一为 log(max(lib,1))。

**H3 修正主张**(这正是「不复现也是发现」的那一半):混杂量级由 sd(log lib) 驱动的**排序律**跨队列
成立;但 r(sd) 不是线性而是在 sd≈1 附近饱和(LYMPH 实测平台 ≈0.24–0.39,视面板/口径),饱和高度
受面板稀疏度压制(LYMPH 官方面板 ≤1 计数占 61–88% > HER2ST 59%)。

背景 spot 的贡献(P1−P2 之差):all_spots 比 in_tissue 高 +0.061(737)~+0.085(hest50)中位;
NCBI684(38.8% 背景)单样本差 +0.137(cp10k,hest_var50 面板;trainHVG737 面板为 +0.163)。方向与机制完全按预注册剧本走,幅度约为预测的一半
(预测 0.19,因为线性外推同时高估了 all_spots 端)。

## 4. H4 天花板(修正估计量:二项对分 + Spearman-Brown + sqrt)与 H5 r_nbr

来源:`lymph_audit_ceiling.csv`(rtech_med_mine = 独立重实现,rtech_med_stnull = stnull 0.2.0
`ceiling.r_tech`,含全库 rest 列对分;两者逐行差 ≤0.008)。r_nbr = 6 近邻真值均值(像素坐标 kNN)。

| 条件(4 样本中位) | r_tech | r_nbr(中位 r) |
|---|---|---|
| hest_var50, all, log1p:raw(**官方评测条件**) | **0.539** | 0.349 |
| hest_var50, all, cp10k:panel | 0.488 | 0.289 |
| hest_var50, in_tissue, cp10k:panel | 0.416 | 0.177 |
| trainHVG737, all, cp10k:panel(HER2ST 可比) | **0.522** | 0.282 |
| trainHVG737, in_tissue, cp10k:panel | 0.446 | 0.206 |

- HER2ST r_tech 中位 0.51 → LYMPH 同口径 0.446–0.522:**测量天花板同样在 ~0.5,复现**。
  逐样本范围 0.33–0.78(NCBI682 最高 0.67–0.78,又是深度最深样本)。
- 官方评测条件下:文库零模型 0.315 / 天花板 0.539 → **零像素已达天花板的 ~58%**;排行榜最好的
  图像模型 0.284(mean-r)连零模型都没过。
- H5:r_nbr 官方条件 mean-r 逐样本 0.356/0.553/0.331/0.457(均值 0.424),**4/4 高于全部已发表模型**。

## 5. stnull 0.2.0 实战(nulls-only 模式,dogfood)

来源:`lymph_stnull_dogfood.csv`、`lymph_stnull_dogfood_notes.txt`。调用:
`nulls_only(Y, space=..., section=sec, lib_size=lib, coords=xy, coord_kind="micron", counts=C, lib_size_full=lib, patient=pat, n_perm=50)`,
4 个配置(2 口径 × 2 spot 集)各 8–11 秒跑完,无报错。

**数字对齐**:
- `depth_null`(cv_within_section)vs 参考实现 med_r_cv:16/16 行差 ≤**0.0013**(不同随机折划分的
  噪声量级)。无 >0.01 的分歧。
- `r_tech` 聚合值 0.492(cp10k/all)、0.416(cp10k/in_tissue)与参考逐样本中位完全一致。
- `r_nbr`:stnull(micron kNN,**k=4**)比参考(k=6)低 0.015–0.039,系近邻数定义差,非 bug;
  已核对同 k 时一致。

**工具在真实外部数据上暴露的问题(0.2.1 输入)**:
1. **块置换在真实 Visium 上全军覆没**:4/4 节「torus embedding covered <50% of spots」回落 FREE
   置换,ledger 有记录、但结果是**在最主流平台上块置换地板事实上不可用**(像素坐标按 pitch 吸附
   对六角格失败)。0.2.1 应支持 Visium 六角格的原生吸附(或用 array_row/col)。
2. **grid 模式假设方格**:Visium `array_row/array_col`(六角,奇偶交错)rook 命中率 0% →
   `_grid_plausibility` 正确识别并自动切 micron(探针 B),提示信息清楚。但「spacing 1.41」这种
   报文对 Visium 用户等于说"你的官方网格坐标不是网格",宜在报文里点名 hex/Visium 情形。
3. 探针 A(像素坐标误标 grid)同样被正确拦截并自动切换——该防护在真实数据上有效。
4. `lib_size has non-positive entries; clamped to 1` 正确命中 NCBI684 的 0 计数 spot。
5. `LibSizeLooksLikePanelSum` 未误报(传的是真全库 lib)。
6. r_nbr 的 k(micron 模式固定 4)未在 `audit()` 签名暴露;建议 0.2.1 暴露 `nbr_k=`。
7. cp10k:panel 口径下面板行和为 0 的 spot(本数据最多 738 个/样本)工具无从感知我们置 0 的规则;
   建议 audit() 对「y_true 中整行常数的 spot」计数并写进 ledger。

## 6. normalize_adata 文档≠代码,三方对齐确认

1. **代码**:`methods\gen4_hest\src\hest\bench\st_dataset.py:55-79` —— docstring 写
   "Normalize each spot by total gene counts + Logarithmize each spot",函数体只有
   `sc.pp.log1p(filtered_adata)`,无任何 `normalize_total`。`benchmark.py:96` 的
   `normalize=True` 走的正是它(勿与 `hest/utils.py:396` 的同名函数混淆,那个真做 CP 归一但
   bench 不调)。
2. **数据**:LYMPH_IDC 四个 h5ad 的 X 为原始整数计数(min=1 非零值、行和=obs/total_counts,
   `lymph_inspect.md` 已逐样本验证)。
3. **合成结论**:HEST-bench 在本数据集上的实际评测目标 = **log1p(原始计数),无任何深度归一**,
   目标向量与 log(lib) 单调耦合 → 本报告 §1 实测该口径把文库零模型抬到 0.315/0.337(中位),
   比 cp10k:panel 高 +0.010~+0.114。docstring 所声称的归一化如果真的存在,排行榜数字会系统性
   变化(方向:下降,幅度约为本表 raw−cp10k 列差)。已可写进 AUDIT_REPORT。

## 7. 已知限制

- n=4 患者:H3 的相关只能当 HER2ST 拟合关系的外部验证点,不能重估斜率;不报 p 值。
- 零模型读测试 spot 真实 lib,是审计基线而非可部署预测器(与 HER2ST 侧同一框架)。
- HEST 排行数字是 Ridge+PCA256 探针、官方折,与我们同折同指标同口径,但特征侧不同实现细节
  (alpha 解析设定)未重跑,引用自 README(03.04.26)。
- 排行对照里 mean-r 与 med-r 已分列,勿混读;所有跨样本聚合 ±0.02–0.05。

# A题：扰流条件下的超声波流量计 — 完整建模分析

## 目录

1. [问题1：超声波流量计基础模型](#一问题1超声波流量计基础模型)
2. [问题2：无扰流多声道流量估计模型](#二问题2无扰流多声道流量估计模型)
3. [问题3：扰流剖面识别与补偿](#三问题3扰流剖面识别与补偿)
4. [问题4：最终补偿模型](#四问题4最终补偿模型)
5. [模型评价与指标分析](#五模型评价与指标分析)
6. [参考文献](#六参考文献)

---

## 一、问题1：超声波流量计基础模型

### 1.1 时差法流速公式

声波在流动水中传播，相对管壁速度需叠加水流拖曳。设声道长度 $L$，与管轴夹角 $\theta$，声速 $c$，轴向流速 $v$：

$$t_+ = \frac{L}{c + v\cos\theta}, \qquad t_- = \frac{L}{c - v\cos\theta}$$

$\Delta t = t_- - t_+$，通分后与 $t_+t_-$ 相除，精确消去声速 $c$：

$$v = \frac{L}{2\cos\theta} \cdot \frac{\Delta t}{t_+ \cdot t_-}$$

该式不含声速 $c$，不受水温影响。$\cos\theta = 1/\sqrt{2}$（附件7，等效 45° 安装角）。$v$ 表示声路上的轴向平均速度 $\bar{v}^{\text{path}}$，非截面点速度。

附件1 chord0–chord4 为窗口级声道等效观测量（由附件6逐秒原始时间差经附件7线性映射得到，量纲等效于 m/s）。

体积公式：窗口持续时间 $T$，等效截面积 $A = 0.13138$ m²（附件7，对应内径 409 mm）：

$$V = A \cdot \bar{v}_A \cdot T$$

当使用窗口内时间平均速度时成立。

### 1.2 多声道积分原理

管道截面二维积分的严格起点为：

$$Q = \iint_A u(x,y)\,dA$$

定义高度 $y$ 处的横向弦平均速度：

$$\bar{u}_x(y) = \frac{1}{2\sqrt{R^2-y^2}} \int_{-\sqrt{R^2-y^2}}^{\sqrt{R^2-y^2}} u(x,y)\,dx$$

令 $t = y/R$，$A = \pi R^2$，截面平均速度为：

$$\bar{u}_A = \frac{Q}{A} = \frac{2}{\pi} \int_{-1}^{1} \bar{u}_x(Rt)\sqrt{1-t^2}\,dt$$

五条声道在固定高度位置给出弦平均速度的离散样本，截面平均速度通过加权和逼近：

$$\bar{u}_A \approx \sum_{i=0}^{4} w_i \cdot \text{chord}_i$$

$w_i$ 为无量纲积分系数，将有限条声路测得的弦平均速度转换为截面平均速度。

### 1.3 声道权重

附件7提供五套权重。三者需明确区分：

- **标准五点 Gauss-Jacobi 求积**（$\alpha=\beta=1/2$）：权函数 $\sqrt{1-t^2}$ 属于 Jacobi 权函数族。标准节点为 $t_i = \{-0.866, -0.5, 0, 0.5, 0.866\}$，归一化权重为 $\{1/12, 1/4, 1/3, 1/4, 1/12\}$。但实际仪表声道位置（$\pm 0.727, \pm 0.266, 0$）与标准节点不同。

- **OWICS**：在实际声道位置上通过加权 Lagrange 插值积分得到：
  $$w_i^{\text{OWICS}} = \frac{2}{\pi} \int_{-1}^{1} L_i(t)\sqrt{1-t^2}\,dt$$
  其中 $L_i(t)$ 为过实际声道位置的 Lagrange 基函数。权重和为 $1.0$：$[0.2212, 0.1122, 0.3332, 0.1122, 0.2212]$。

- **Phys6**：附件7在积分模型基础上给出的工程修正系数，权重和为 $0.993$：$[0.2099, 0.1532, 0.2668, 0.1532, 0.2099]$。附件7未公开权重和小于 $1.0$ 的具体修正依据。

声道权重不是截面环带面积占比——五条声道对应五条横向弦线，非五个同心圆环；OWICS 权重来自插值积分，可在部分区域取负值或大于 1；Phys6 权重和不为 1，不可能严格解释为面积比例。

### 1.4 四套积分方案对比

在全量 159 窗口上对比附件1预计算的四种积分方案：

| 方法 | MAE | 系统偏差 | 最大绝对相对误差 |
|------|:---:|:---:|:---:|
| Phys6 | 0.181% | −0.136% | 0.495% |
| Lagrange | 0.184% | −0.008% | 0.644% |
| OWICS | 0.506% | +0.506% | 1.001% |
| 等权 | 0.637% | +0.637% | 1.084% |

Phys6 和 Lagrange 精度相当。OWICS 是基于实际声道位置的理论积分方案，物理上自洽；其在当前数据上呈现正偏差，说明理想插值积分假设与实际声路响应间存在系统差异。等权法忽略声道积分贡献差异，误差最大。上述结果为 159 窗口上的描述性评价，不代表某种权重在所有流场条件下普遍最优。

---

## 二、问题2：无扰流多声道流量估计模型

### 2.1 模型定位

D0 仅 1 天 10 窗口、5 个流量点（40–80），数据驱动的参数拟合不可行。模型参数须由数学推导确定。实际采用附件7给出的固定积分参数，不由 D0 数据拟合。

### 2.2 权重推导

在实际声道位置 $\{0, \pm 0.266, \pm 0.727\}$ 上，通过加权 Lagrange 插值积分推导 OWICS 权重。对每声道位置 $t_i$ 构造 Lagrange 基函数 $L_i(t)$：

$$w_i^{\text{OWICS}} = \frac{2}{\pi} \int_{-1}^{1} L_i(t)\sqrt{1-t^2}\,dt$$

数值积分结果与附件7 `weight_owics` 列一致（偏差 < $10^{-5}$），验证了代码实现的正确性。

### 2.3 模型形式

$$V = A \cdot T \cdot \sum_{i=0}^{4} w_i^{\text{Phys6}} \cdot \text{chord}_i$$

选取 Phys6 权重作为最终模型参数。在全量 159 窗口的五套权重官方指标对比中，Phys6 取得组通过 5/30，$u_{\text{nor},L}=0.441\%$，$u_{\text{nor},r}=0.122\%$，$u_{\text{nor},d}=0.297\%$，四项均最优或并列最优。

| 符号 | 值 | 来源 | 类型 |
|------|-----|------|------|
| $A$ | 0.13138 m² | 附件7，等效截面积 | 固定物理常数 |
| $T$ | 逐窗口不同 | 附件1 `duration_s` | 输入变量 |
| chord$_0$–chord$_4$ | 逐窗口不同 | 附件1，声道等效线平均流速 | 输入变量 |
| $w_0$–$w_4$ | 0.210, 0.153, 0.267, 0.153, 0.210 | 附件7 Phys6 工程修正系数 | 固定积分参数 |

### 2.4 D0 无扰流表现

Phys6 在 D0（10 窗口）上 MAE $0.389\%$，偏差 $-0.389\%$，SD $0.051\%$。误差近似为统一比例偏差，流量点间变化小。该偏差可能来自权重总尺度、截面积参数或 chord 映射等多种因素，仅凭 10 个窗口无法唯一确定来源。

### 2.5 与问题1方法对比

| 方法 | MAE | 系统偏差 | 物理基础 |
|------|:---:|:---:|------|
| Phys6（本文） | 0.181% | −0.136% | 工程修正积分系数 |
| OWICS | 0.506% | +0.506% | 实际声道位置理论积分 |
| Lagrange | 0.184% | −0.008% | Lagrange 型插值积分 |
| 等权 | 0.637% | +0.637% | 无 |

---

## 三、问题3：扰流剖面识别与补偿

### 3.1 扰流敏感特征

Cohen's d 评估 16 个候选特征的 D0 vs 扰流判别力。`profile_swirl`（$d=39.7$）和 `profile_ab_abs`（$d=33.1$）判别力最强，在 D0 与扰流间分布无重叠。

双阈值 OR 规则：$|\text{profile\_ab\_abs}| > 0.0386$ 或 $|\text{profile\_swirl}| > 0.0226$。阈值由 D0 最大值 $+3\sigma$ 确定。样本内检测正确 159/159。

### 3.2 扰流聚类

12 维特征（5 归一化 chord + 5 AB 差异 + profile_swirl + profile_ab_abs），D0 参照 Z-score 标准化，PCA 降至 2 维（方差 94.6%），Ward 层次聚类。

K=2 轮廓系数 0.617，最优。A 类 = {D1, D2, D5, D7}，B 类 = {D3, D4, D6, D8}。两类核心差异为 profile_top_bottom 正负：A 类剖面上偏（扰流件装在上半管），B 类下偏。

### 3.3 在线识别

不依赖 `disturbance_id`：第一层双阈值检测扰流存在；第二层 12 维特征 → D0 标准化 → PCA 降维 → Mahalanobis 距离分类（A/B）。

留一日期交叉验证分类正确率 32/149（21.5%）——多数日期仅含一种扰流，训练集中未见过该类型时无法正确分类。这不是分类方法缺陷，而是日期与扰流编号完全混杂所致。

### 3.4 分流量点补偿

ANOVA 确认扰流误差随流量点显著变化（$p < 10^{-6}$）。两类按流量点查表补偿：

$$\hat{V} = V_{\text{base}} \times (1 + \delta_{c, p})$$

Phys6 基线 MAE 0.181%，扰流均值归零补偿后 MAE 0.118%，组通过 8/30。最大组内 SD 仍约 0.123%，说明常数补偿只能处理类型和流量点层面偏差，不能消除同组窗口间波动。

---

## 四、问题4：最终补偿模型

### 4.1 模型架构

问题4采用 Phys6 物理基线 + 极端随机树残差补偿的混合架构：

$$\hat{V}_i = \underbrace{A \cdot T_i \cdot \sum_{j=0}^{4} w_j \cdot \text{chord}_{ij}}_{\text{Phys6 物理基线 } V_i^{(0)}} \cdot\; \exp\!\left(\underbrace{\frac{1}{M}\sum_{m=1}^{M} T_m(\boldsymbol{x}_i)}_{\text{ET 残差预测 } \hat{r}_i}\right)$$

第一层物理积分负责主体流量计算；第二层 ExtraTrees 对对数残差 $r_i = \log(V_i^{\text{std}}/V_i^{(0)})$ 进行非线性补偿。模型使用 28 维在线特征（5 chord + 5 AB + 6 profile + 6 dynamic + 3 zero + 3 auxiliary）。

### 4.2 验证方法

嵌套留一日期交叉验证：外层每次留出一个完整日期为测试集，内层仅利用其余日期选择 ExtraTrees 的叶节点样本数、最大树深、随机特征比例和树数量。各折缺失值中位数仅由对应训练集计算。固定 $\eta=1$（残差收缩搜索未改善外层结果）。

### 4.3 模型比较

在相同嵌套留一日期验证框架下，七种补偿模型对比（13–28 维特征）：

| 模型 | 组通过 | $u_{\text{nor},L}$ | $u_{\text{nor},r}$ | $u_{\text{nor},d}$ |
|------|:---:|:---:|:---:|:---:|
| Phys6 基线 | 5/30 | 0.441% | 0.122% | 0.297% |
| Ridge 残差 | 5/30 | 1.417% | 0.147% | 0.961% |
| RBF-SVR 残差 | 4/30 | 0.338% | 0.155% | 0.348% |
| GBRT 残差 | 7/30 | 0.154% | 0.151% | 0.266% |
| 动态声道权重（对称） | 8/30 | 0.294% | 0.123% | 0.309% |
| **ET(13d)** | **10/30** | **0.189%** | 0.121% | 0.286% |
| **ET(28d)** | **10/30** | 0.203% | **0.114%** | **0.280%** |

ET(28d) 为精度最优模型。13 维简化模型（profile+dyn+rate）同为 10/30 且 $u_L$ 更优，可作为论文简化对照。

### 4.4 特征消融

9 组特征组合的消融实验表明：profile+dyn+rate（13 维）与全部 28 维同为 10/30；AB 不对称特征和零点特征未带来稳定增益；动态特征（plateau_cv、启停比例等）是唯一系统性改善跨日期泛化的特征组。逐秒时序特征（25 维）单独或合并使用均未超越 13 维窗口特征。

---

## 五、模型评价与指标分析

### 5.1 评价框架

问题四要求"尽可能满足五项指标"，非强制全部达标。评价采用改善幅度叙事：逐组判断组平均误差与组内标准差是否同时满足 $|\bar{e}_g| \le 0.2\%$ 且 $SD_g \le 0.040\%$，并计算 $u_{\text{nor},L}$、$u_{\text{nor},r}$、$u_{\text{nor},d}$ 三项全局指标。

### 5.2 五项目标完成情况

| 目标 | Phys6 基线 | ET(28d) 补偿后 | 阈值 | 改善 |
|------|:---:|:---:|:---:|:---:|
| 组级双阈值通过 | 5/30 (16.7%) | 10/30 (33.3%) | — | 翻倍 |
| $u_{\text{nor},L}$ | 0.441% | 0.203% | <0.036% | ↓54% |
| $u_{\text{nor},r}$ | 0.122% | 0.114% | <0.040% | ↓6.5% |
| $u_{\text{nor},d}$ | 0.297% | 0.280% | <0.115% | ↓5.7% |
| MAE（补充） | 0.181% | 0.116% | — | ↓36% |

### 5.3 评价分组结构

30 个有效评价组中，12 组为单一扰流状态，18 组同时含两种扰流类型。组内 SD 同时包含同扰流类型内窗口波动和不同扰流类型间补偿偏差。混合组对模型提出了更高要求——须依据在线剖面特征区分同一日期同一流量点下的不同扰流状态。

### 5.4 诚实评价

残差补偿主要改善了组间系统偏差和流量点间线性度，组通过数翻倍。但重复性指标 $u_{\text{nor},r}$ 在全部七种模型中始终处于 0.114%–0.155%，远超 0.040% 阈值。28 维 ET 取得当前最低值 0.114%，仅比 Phys6 改善 6.5%。

当前模型最擅长修正组间和流量点间系统偏差，最薄弱的是降低同工况重复测量离散性及复杂扰流影响。五项目标中没有任何模型同时满足全部要求。该结果表示，现有窗口级观测变量能够较好地解释工况间系统偏差，但对同一评价组内短时波动和不同扰流窗口间差异的解释能力有限。

三个全局指标与阈值的差距分别为：$u_L \approx 5.6\times$，$u_r \approx 2.9\times$，$u_d \approx 2.4\times$。该差距来自 8 个日期、159 个窗口的有限样本和日期-扰流状态的混杂，不代表该问题存在不可突破的理论精度上限。

---

## 六、参考文献

[1] Zheng D, Zhang P, Xu T. Improved numerical integration method for flowrate of ultrasonic flowmeter based on Gauss quadrature for non-ideal flow fields. *Flow Measurement and Instrumentation*, 2015, 41: 28–35.

[2] Tresch T, Lüscher B, Staubli T, Gruber P. Presentation of optimized integration methods and weighting corrections for the acoustic discharge measurement. *IGHEM Conference*, Milano, 2008.

[3] Roman V, Matiko F, Kutsan Y. Software for calculating the location coordinates and weighting coefficients of acoustic paths of ultrasonic flow meters. *Energy Engineering and Control Systems*, 2022, 8(2): 144–150.

[4] ISO 12242:2012. Measurement of fluid flow in closed conduits — Ultrasonic transit-time meters for liquid.

[5] Salami LA. Application of a computer to asymmetric flow measurement in circular pipes. *Trans. Inst. Meas. Control*, 1984, 6(5): 261–272.

[6] Cordova ML, Lederer T. A new approach to improve reproducibility of ultrasonic flow meters. *FLOMEKO*, 2013.

[7] Papathanasiou P et al. Flow disturbance compensation calculated with flow simulations for ultrasonic clamp-on flowmeters. *Flow Measurement and Instrumentation*, 2022, 85: 102164.

[8] Ton V. A novel approach to multi-path ultrasonic transit time flow meter based on measurement model analysis to improve accuracy. *Flow Measurement and Instrumentation*, 2023, 91: 102352.

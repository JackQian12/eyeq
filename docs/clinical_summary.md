# EyeQ 临床说明文档
# EyeQ Clinical Reference Document

**适用读者 / Intended Readers：** 眼科医生、视光师、临床研究人员  
**Ophthalmologists · Optometrists · Clinical Researchers**

**文件性质 / Document Type：** 临床参考说明（非注册申报文件）  
**版本 / Version：** 1.0 &emsp; **日期 / Date：** 2026-02-24

> ⚠️ **重要声明 Important Notice**  
> EyeQ 目前处于概念验证（PoC）阶段，**尚未取得任何医疗器械注册证**，所有输出结果**仅供健康参考，不构成临床诊断**。  
> EyeQ is currently a research prototype. It has **not received any medical device clearance**. All outputs are for **informational purposes only and do not constitute a clinical diagnosis.**

---

## 目录

1. [设备概述](#1-设备概述)
2. [测量原理与临床对应关系](#2-测量原理与临床对应关系)
3. [输出指标的临床解读](#3-输出指标的临床解读)
4. [风险分级说明](#4-风险分级说明)
5. [局限性与干扰因素](#5-局限性与干扰因素)
6. [与金标准检查的关系](#6-与金标准检查的关系)
7. [适用场景建议](#7-适用场景建议)
8. [参考文献](#8-参考文献)

---

## 1. 设备概述

EyeQ 是一款搭载**内向摄像头**（朝向用户眼部）的智能眼镜原型系统，通过计算机视觉算法**持续、非接触、非侵入性地**监测用户的眨眼行为，并以此间接评估泪膜稳定性风险。

### 工作原理简述

```
眼镜内置摄像头（60 fps）
        ↓
MediaPipe Face Mesh 提取眼部 486 个三维关键点
        ↓
计算双眼眼睛纵横比（EAR，Eye Aspect Ratio）时间序列
        ↓
状态机检测每次眨眼事件（完全/不完全/长闭眼）
        ↓
滑动窗口（60 秒）统计行为指标 → 综合风险评分
```

**核心特点：**
- 无需荧光素染色或任何接触性操作
- 可在日常佩戴中连续监测（分钟至小时级）
- 适合门诊前筛查、居家健康监测、临床研究数据采集

---

## 2. 测量原理与临床对应关系

EyeQ 无法直接测量泪膜，而是通过**眨眼行为模式**间接推断泪膜稳定性。以下说明各行为指标与临床检查项目的对应关系。

### 2.1 眨眼间期（IBI）→ 非侵入性泪膜破裂时间（NIBUT）

| 临床概念 | EyeQ 对应指标 | 两者关系 |
|---|---|---|
| NIBUT（非侵入性泪膜破裂时间）| 眨眼间期均值（IBI mean） | 间接代理（proxy），非等价 |
| TBUT（荧光素染色泪膜破裂时间）| — | 本系统不涉及 |

**生理逻辑：**  
泪膜完整时，用户无需频繁眨眼；泪膜破裂产生的异物感或干涩感会触发**反射性眨眼**。因此，两次相邻眨眼之间的时间间隔（IBI）可视为"泪膜至少能维持的时间"的行为性标志物。

$$
\text{IBI（行为）} \lesssim \text{NIBUT（生理）}
$$

IBI 通常**低于**真实 NIBUT，原因是：
1. 任务导向的**有意识抑制**（如专注阅读）可延长 IBI，使其超过真实 NIBUT
2. 部分眨眼为**习惯性（非反射性）**，与泪膜无关

> **临床意义：** IBI 均值 < 3 秒或 > 8 秒均提示异常；结合频率和不完全眨眼比例，可作为干眼筛查的辅助信号。

---

### 2.2 不完全眨眼比例（IBR）→ 睑缘腺功能与角膜暴露

| 临床概念 | EyeQ 对应指标 |
|---|---|
| 不完全眨眼 / 眼睑闭合不全 | 不完全眨眼比例（IBR, Incomplete Blink Ratio） |
| 睑板腺功能障碍（MGD）间接标志 | IBR > 40% |

**什么是不完全眨眼？**  
眨眼时上眼睑下降行程不足，未能完全覆盖角膜（通常完成度 < 2/3），即为不完全眨眼。

**临床危害（McMonnies, 2007；Tian, 2021）：**

1. **泪液涂布不均**：上眼睑是将泪膜均匀涂抹角膜的"刮板"，不完全眨眼导致下方角膜区域（暴露区）缺乏涂布。
2. **睑板腺挤压不足**：睑板腺的脂质分泌依赖完整的眨眼压力，不完全眨眼减少脂质输出，加速泪膜蒸发。
3. **角膜点染高发区**：不完全眨眼患者的下方角膜（距眼睑最远处）点染检出率显著高于正常人。

**EyeQ 的检测方式：**  
系统通过 EAR（眼睛纵横比）时间序列，检测每次眨眼的最低 EAR 值。若最低 EAR 未越过完全闭合阈值（0.20），但进入不完全闭合区间（0.20–0.23），则记录为不完全眨眼。

---

### 2.3 眨眼频率（BR）→ 泪液分布与神经调控状态

| 临床意义 | 眨眼频率范围 |
|---|---|
| 正常（觉醒、休息状态）| 15–25 次/分钟 |
| 低频（专注屏幕、抑制状态）| 5–12 次/分钟 |
| 高频代偿（眼表刺激）| > 25 次/分钟 |
| 帕金森病（基底节多巴胺减少）| < 10 次/分钟 |
| 迟发性运动障碍 / 睑痉挛 | > 30 次/分钟 |

低眨眼频率在眼科的主要意义是**泪液涂布不足**，与数字眼疲劳（Digital Eye Strain）高度相关（Rosenfield, 2016）。

---

### 2.4 IBI 变异系数（IBI-CV）→ 眨眼节律稳定性

IBI 的变异系数（CV = 标准差 / 均值）反映眨眼节律的规律程度。

| CV 水平 | 临床关联 |
|---|---|
| < 0.3 | 规律，正常 |
| 0.3–0.5 | 轻度不规律，可能受任务切换或情绪波动影响 |
| > 0.5 | 高度不规律，提示眨眼反射调控异常或显著任务干扰 |

---

## 3. 输出指标的临床解读

EyeQ 在每 60 秒滑动窗口内输出以下指标（需至少 3 次有效眨眼）：

### 指标速查表

| 指标 | 单位 | 正常参考范围 | 异常提示 |
|---|---|---|---|
| **眨眼频率（BR）** | 次/分钟 | 15 – 25 | < 10：显著低频；> 30：代偿性高频 |
| **不完全眨眼比例（IBR）** | % | < 30% | ≥ 40%：MGD 风险；≥ 60%：高度怀疑眼睑功能障碍 |
| **IBI 均值**（估算 NIBUT） | 秒 | 3 – 7 s | < 2 s：过度眨眼；> 8 s：泪膜可能过稳定或有意抑制 |
| **IBI 标准差** | 秒 | < 2.0 s | — |
| **IBI 变异系数（CV）** | 无量纲 | < 0.4 | > 0.6：节律显著紊乱 |
| **综合风险评分** | 0 – 100 | 0 – 30（低） | 31–60 中风险；61–100 高风险 |

> **注意：** 上述参考范围来自健康觉醒成人在**自然休息状态**下的数据。专注屏幕工作时，正常人 BR 可降至 4–9 次/分钟，不应直接套用上表。上下文环境（任务类型）对解读至关重要。

---

## 4. 风险分级说明

EyeQ 的综合风险评分（0–100）由四项子评分等权求和：

| 子项 | 满分 | 评估维度 |
|---|---|---|
| $S_{\text{BR}}$ | 25 | 眨眼频率偏离正常范围的程度 |
| $S_{\text{IBR}}$ | 25 | 不完全眨眼比例（与 MGD/干眼相关性最强）|
| $S_{\text{IBI}}$ | 25 | 眨眼间期（间接估算泪膜维持能力）|
| $S_{\text{CV}}$ | 25 | 眨眼节律稳定性 |

### 风险等级临床解读

| 评分 | 等级 | 临床解读 | 建议处理 |
|---|---|---|---|
| **0 – 30** | 🟢 低风险 | 眨眼行为模式正常，泪膜状态良好的可能性高 | 继续观察，无需干预 |
| **31 – 60** | 🟡 中风险 | 存在一项或多项眨眼行为异常，提示眼表可能处于亚临床干眼状态 | 建议进行标准干眼问卷（OSDI/SPEED）及裂隙灯检查 |
| **61 – 100** | 🔴 高风险 | 多项眨眼指标异常，高度提示泪膜不稳定，干眼症可能性较大 | 建议专科就诊，完整干眼评估（NIBUT、泪液分泌、睑板腺成像）|

**重要提示：** 单次测量结果受当前任务状态影响较大。建议参考**多次测量的趋势**（如连续 3 天、每天测量 20 分钟以上），而非单次结果做出临床判断。

---

## 5. 局限性与干扰因素

### 5.1 系统性局限

| 局限 | 说明 |
|---|---|
| **无法测量泪液量** | EyeQ 不含泪液分泌测试（Schirmer 试验）能力，无法评估泪液水量不足型干眼（Aqueous-deficient DED）|
| **无法测量泪液渗透压** | 泪液渗透压是干眼诊断的重要生化指标，本系统不涉及 |
| **无法测量睑板腺形态** | 睑板腺缺失或萎缩需红外睑板腺成像（meibography）|
| **IBI ≠ NIBUT** | 两者相关但不等价（见第 2.1 节），不可直接替代临床 NIBUT 测量 |
| **仅评估功能性干眼信号** | 结构性病变（角膜上皮损伤、角膜点染）无法被本系统发现 |

### 5.2 拍摄与算法干扰因素

| 干扰因素 | 对指标的影响 | 应对建议 |
|---|---|---|
| **强光 / 逆光** | MediaPipe 关键点定位误差增大，EAR 噪声上升，可能产生假性眨眼 | 避免在强背光环境下使用或标注测量条件 |
| **眼镜镜片（厚框 / 厚镜片反射）** | 遮挡关键点，EAR 计算不可靠 | PoC 阶段建议摘除眼镜使用（眼镜端嵌入摄像头后可缓解）|
| **单睑 / 小眼裂个体** | 基线 EAR 低（0.22–0.28），标准阈值（0.20）可能漏检部分完全眨眼 | 建议为此类用户进行个性化校准 |
| **专注性任务（阅读/游戏）** | BR 被有意识抑制，IBI 延长，评分可能虚高 | 解读时充分考虑测量时的任务背景 |
| **困倦 / 微睡眠** | 出现 PROLONGED（长闭眼）事件，BR 下降 | 系统已将长闭眼（>1.5 s）单独标注，不计入频率 |
| **点眼药水（人工泪液）** | 短期改善泪膜，BR 可能降低，IBR 改善 | 可作为用药前后对照的观察指标 |
| **干燥环境（湿度 < 40%）** | 加速泪膜蒸发，IBI 可能缩短 | 建议记录测量时环境湿度 |

### 5.3 人群适用性限制

本系统在以下人群中尚未经过验证，解读结果需额外谨慎：

- 儿童（< 12 岁；眨眼模式与成人存在差异）
- 严重眼睑外翻 / 内翻患者
- 面神经麻痹（面瘫）导致眼睑闭合障碍的患者
- 正在使用影响神经系统药物的患者（抗精神病药、多巴胺能药物等）

---

## 6. 与金标准检查的关系

下表说明 EyeQ 指标与眼科临床金标准的关系，帮助临床医师理解其在诊疗流程中的定位。

| 临床检查项目 | 金标准 | EyeQ 的对应能力 | EyeQ 的定位 |
|---|---|---|---|
| 泪膜破裂时间（TBUT）| 荧光素染色 + 裂隙灯 | IBI 均值（间接代理）| 筛查参考，不可替代 |
| 非侵入性泪膜破裂时间（NIBUT）| Keratograph / Oculus Keratograph 5M | IBI 均值（相关性中等）| 筛查参考，不可替代 |
| 泪液分泌量（Schirmer）| 滤纸条 5 分钟 | 无对应 | — |
| 睑板腺成像（Meibography）| 红外透照 | 无对应（IBR 为功能性间接指标）| — |
| 不完全眨眼评估 | 裂隙灯 / 高速摄像 | IBR ✅（核心功能，精度较高）| 有一定替代价值 |
| 干眼问卷（OSDI / SPEED）| 患者自报 | 无对应 | 建议配合问卷使用 |
| 角膜荧光素点染 | 裂隙灯 | 无对应 | — |

### 推荐工作流程（建议）

```
1. EyeQ 筛查（日常佩戴，连续 ≥20 分钟）
         ↓
2. 若风险评分 ≥31 或 IBR ≥30%
         ↓
3. 完成 OSDI / SPEED 问卷
         ↓
4. 眼科就诊：NIBUT、Schirmer、睑板腺成像、角膜染色
         ↓
5. 临床诊断 + 治疗方案制定
```

EyeQ 的价值在于**降低就诊门槛**（无需到院即可持续监测），以及在治疗随访中提供**客观的行为指标变化趋势**，而非替代临床检查。

---

## 7. 适用场景建议

### 适合使用 EyeQ 的场景 ✅

| 场景 | 说明 |
|---|---|
| **职业性长时间屏幕使用人群筛查** | IT 工作者、游戏玩家、设计师等高风险人群的日常干眼风险自我监测 |
| **干眼治疗随访辅助** | 人工泪液、热敷、脂质层治疗等的疗效观察（对比用药前后 IBR、IBI 变化趋势）|
| **流行病学研究数据采集** | 长时程、大样本的眨眼行为数据获取，弥补一次性临床检查的时间点局限 |
| **干眼高危人群预警** | 糖尿病（角膜神经病变）、帕金森、屈光手术前后患者的动态监测 |
| **远程/居家健康管理** | 无法频繁到院的患者（老年人、慢性病患者）的日常眼部健康参考 |

### 不适合替代临床检查的场景 ❌

| 场景 | 原因 |
|---|---|
| 干眼症的确诊与分型 | 需要泪液渗透压、TBUT、Schirmer、睑板腺成像等多维度临床数据 |
| 眼表疾病的治疗方案决策 | 医疗决策不可基于单一行为性筛查工具 |
| 角膜、结膜、睑板腺结构性病变评估 | 本系统仅反映功能性行为指标 |
| 儿童及特殊人群的眼部疾病诊断 | 尚无该人群的系统验证数据 |

---

## 8. 参考文献

1. Soukupová T, Čech J. *Real-Time Eye Blink Detection using Facial Landmarks.* 21st Computer Vision Winter Workshop (CVWW), 2016.

2. Doughty MJ. *Consideration of three types of spontaneous eyeblink activity in normal humans.* Optom Vis Sci. 2001; 78(9): 712–725.

3. Tian L, Qu JH, Zhang XY, Sun XG. *Repeatability and reproducibility of noninvasive keratograph 5M measurements in patients with dry eye disease.* J Ophthalmol. 2016; 2016:8013621.

4. Tian L, Cao K, Wang HZ, et al. *Incomplete blinking as a risk factor for dry eye disease among young office workers: a cross-sectional study.* Cornea. 2021; 40(9): 1193–1199.

5. McMonnies CW. *Incomplete blinking: exposure keratopathy, lid wiper epitheliopathy, dry eye, refractive surgery, and dry contact lenses.* Cont Lens Anterior Eye. 2007; 30(1): 37–51.

6. Rosenfield M. *Computer vision syndrome (a.k.a. digital eye strain).* Optometry in Practice. 2016; 17(1): 1–10.

7. Bentivoglio AR, Bressman SB, Cassetta E, et al. *Analysis of blink rate patterns in normal subjects.* Mov Disord. 1997; 12(6): 1028–1034.

8. Mengher LS, Bron AJ, Tonge SR, Gilbert DJ. *Effect of fluorescein instillation on the pre-corneal tear film stability.* Curr Eye Res. 1985; 4(1): 9–12.

9. The Ocular Surface Society (TFOS). *TFOS DEWS II Report: Definition and Classification of Dry Eye Disease.* Ocul Surf. 2017; 15(3): 276–283.

10. Sullivan BD, Crews LA, Messmer EM, et al. *Correlations between commonly used objective signs and symptoms for the diagnosis of dry eye disease.* Invest Ophthalmol Vis Sci. 2014; 55(9): 6032–6040.

11. Evinger C, Manning KA, Sibony PA. *Eyelid movements: mechanisms and normal data.* Invest Ophthalmol Vis Sci. 1991; 32(2): 387–400.

12. Lugaresi C, et al. *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172. 2019.

---

*本文档由 杰克编制，供临床合作研究参考。如需引用数据或开展临床研究合作，请联系项目负责人。*

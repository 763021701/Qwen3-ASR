# 真实域病理口述关键词与短语

供合成数据覆盖用。来源是当前 real raw+denoised 划分：

| 划分 | 清单 | 独立文本 | 病例组 |
|---|---|---:|---|
| train | `data/poc_train_real_raw_denoised/train.jsonl` | 898 | 13229, 13247, 13290, 13373, 13488, Sample1–10（不含 13297/13342） |
| dev | 同目录 `dev.jsonl` | 288 | **13297, 13342**（甲状腺、staple line 主要在这里） |
| test | `raw/POC_test/metadata.jsonl` | 53 | 12009, 12712, 26SS11678, 26SS11731, 26SS12082, 26SS12099 |

机器可读完整表（含每条的 train/dev/test 计数）：[`real_domain_medical_keywords.json`](real_domain_medical_keywords.json)。

计数按**去重后的独立文本**（raw/denoised 同一句只计一次）。不要把测试集原句整段拿去合成。

优先级：

- **test_only**：测试有、当前 train 没有。合成时最该加重。
- **dev_only**：真实 dev 有、train 没有（尤其甲状腺）。
- **train_and_test / train_only**：已有文本覆盖，合成主要用来补声学多样性。

---

## 1. 合成时优先覆盖（测试有、train 基本没有）

这些是当前 real-only 模型最容易漂到错误模板的短语。

### 1.1 冰冻 / 石蜡 / 刮片

- scrape cytology
- cytology smear
- scrape cytology smear and block (A) sampled for frozen and paraffin sections
- frozen and paraffin sections
- paraffin section / paraffin sections
- sampled for frozen and paraffin sections
- background thyroid
- rest of the nodule was all embedded in block (B) for paraffin section

train 里接近的只有 `frozen block of {margin}`，没有 `scrape cytology`，也没有 `frozen and paraffin sections`。

### 1.2 前哨淋巴结缩写

- sentinel node / left sentinel node / sentinel lymph node
- SN 1 / SN 2 / SN1 / SN2
- non-SN
- Submitted fresh were three lymph nodes labelled SN 1, SN 2 and non-SN
- SN {n} measured … and was bisected and all embedded in block ({letter}) for frozen and paraffin sections

train 里接近的是 `right axillary S L N level I number {one,two,three}`（逐字母 S L N），没有 `sentinel`，也没有 `non-SN`。

### 1.3 甲状腺 / 乳房术式

- hemithyroid / left hemithyroid
- hemithyroidectomy / hemithyroidectomy specimen
- lumpectomy / left lumpectomy / lumpectomy specimen
- submitted fresh was a {hemithyroidectomy|lumpectomy} specimen weighing {n} gram

`thyroidectomy` / `total thyroidectomy` 在 **dev（13342）**，不在 train。

### 1.4 开场、定向、染色

- submitted fresh（train 用 `received fresh` / `received in formalin`）
- inner and outer sides were inked black and blue respectively
- superficial and deep halves were inked blue and black respectively
- superior edge was inked yellow
- stitches indicating orientation
- all margins perpendicular
- coronal full slabs of the tumour
- surgical staples / a line of surgical staples
- staple line（**dev**）

### 1.5 测试集解剖/大体词（train 很少或没有）

- parametrium
- endometrial cavity
- across the cornu
- posterior wall
- pleural surface / subpleural / focally puckered
- lung nodule / left lower lobe lung nodule
- anteroposteriorly（train 多为 `antero-posteriorly` 或 `anteriour`）
- translucent light brown
- haemorrhagic cystic change
- homogeneous tan cut surfaces
- whitish nodule / whitish tumour / soft oval nodule
- bulging fibroids / subserosal fibroids
- right ear mast（也可能是 mass 的截断）

### 1.6 标本号模式（不要用测试原号）

测试写法：`12009`、`12712`、`26SS11678`、`26SS11731`、`26SS12082`、`26SS12099`。

合成用**模式**，不要复制这些 ID：

- `{5-digit}` → 读成中文数字或逐位英文数字
- `{2-digit}SS{5-digit}` → `二六 S S 一一六七八` / `26 SS 11678` / `26SS11678` 三种读法都要有

---

## 2. 可直接填空的口述模板

把花括号换成词表里的项即可。同一模板用多种数字读法（阿拉伯、`two`、`兩`、`二`）。

```
Specimen labelled {laterality} {specimen}.
Specimen labelled {specimen} is received in formalin, it consists of ...
Submitted fresh was a {procedure} specimen weighing {n} gram.
Submitted fresh were three lymph nodes labelled SN 1, SN 2 and non-SN.
Received specimen labelled {laterality} {specimen}. It consists of ...
It consists of a {descriptor} piece of {tissue} measuring {a} × {b} × {c} mm in size.
The {structure} measures {a} mm in length and {b} mm in diameter, period.
It measured {a} cm across, {b} cm longitudinally and {c} cm anteroposteriorly.
The {surface} was {descriptor} and was inked {color}.
Inner and outer sides were inked black and blue respectively.
Superficial and deep halves were inked blue and black respectively. Superior edge was inked yellow.
Sectioning shows a {location} {descriptor} nodule measuring {a} cm × {b} mm × {c} mm.
Scrape cytology smear and block ({letter}) sampled for frozen and paraffin sections.
The rest of the nodule was all embedded in block ({letter}) for paraffin section. Total {n} blocks.
{n} paraffin blocks were subsequently taken as follows: (B and C) All embedding the rest of the nodule. (D) Background thyroid. Total {n} 個 blocks.
SN {n} measured {a} cm × {b} mm × {c} mm and was bisected and all embedded in block ({letter}) for frozen and paraffin sections.
Non-SN measured {a} mm × {b} mm × {c} mm and was bisected and all embedded in block ({letter}) for frozen and paraffin sections.
{n} blocks were taken for frozen and paraffin sections as follows (all margins perpendicular):
(A) Superior and inferior margins. (B) Medial margin. (C) Lateral margin. (D) Superficial margin. (E) Deep margin.
Serially sectioned. All embedded in {n} blocks.
Bisected. All embedded in {n} frozen blocks. Block {letter} Frozen block. Block {letter} Rest of tissue.
All fibroids show pink firm tissue on cut surfaces, period. No necrosis or haemorrhage can be identified, period.
The adjacent myometrium is grossly unremarkable, period. The endometrial cavity appears smooth with the endometrium measuring up to {n} mm thick, period.
Sections show {tissue} with mild chronic inflammation. No dysplasia or malignancy is seen. No helicobacter seen, full stop.
{a} × {b} × {c} mm in size.
{a} c m 乘 {b} c m 乘 {c} c m
The specimen weighs {n} gram, period.
End of dictation. Thank you.
```

---

## 3. 词表（按类）

### 3.1 开场 / 框架

specimen labelled, it consists of, submitted fresh, received fresh, received in formalin, submitted was, submitted were, in aggregate, in greatest dimension, in size, weighing, the specimen weighs, representative blocks taken, taken as follows, summary of sections, end of dictation, thank you

口头标点：`period`（测试 12712 几乎每句）、`full stop`（train 胃镜/活检很多）、`comma`、`dash`、`o'clock`

### 3.2 取材动作

all embedded, all embedded in, embedded in block(s), entirely submitted, submitted in toto, bisected, serially sectioned, serial sectioning, section shows, sectioning shows, cut section shows, cut surfaces, frozen block, frozen section, paraffin section(s), scrape cytology, cytology smear, additional block, rest of tissue, rest of the nodule, background thyroid

### 3.3 定向与染色

inked, inked black, inked blue, inked yellow, painted, orientation, stitches, stitch, suture, staple, staples, surgical staples, staple line, wire, all margins perpendicular, superficial and deep halves, superior / inferior / medial / lateral / superficial / deep margin

### 3.4 术式与标本名

**测试缺口：** hemithyroid, hemithyroidectomy, lumpectomy, sentinel node

**train 已有：** hysterectomy, mastectomy, hemicolectomy, salpingo-oophorectomy, oophorectomy, wedge resection, LEEP cone biopsy, uterine curettings, uterine curettage, endometrial polyp(s), cervical polyp

**dev 才有：** thyroidectomy, total thyroidectomy, staple line

### 3.5 `specimen labelled` 后常见名称

直接可当合成主语：

- gallbladder
- uterus / uterus and bilateral tubes and ovaries / uterus and bilateral appendages / uterus and its appendages / uterus and fibroid / uterine fibroids
- left ovarian cyst, left kidney
- left breast, right lateral breast mass
- left oral cavity mass, left maxillary sinus mass, vallecular cyst
- left lower lobe apical segment, left lower lobe lung nodule, left upper lobe, left upper lobe wedge (bulla)
- gastric submucosal tumour, sigmoid colon, sigmoid colon polyp, ascending colon polyp
- LEEP cone biopsy of cervix, liver cyst wall, cervical polyp, endometrial polyp(s), haemorrhoids
- epidural abscess soft tissue, (abdominal) peritonaeal biopsy
- interlobar / hilar / left paratracheal / left pelvic / left para-aortic / pulmonary ligament lymph node
- left hemithyroid, left lumpectomy, left sentinel node
- right ear mast

### 3.6 淋巴结

lymph node(s), sentinel node, sentinel lymph node, axillary, pelvic, para-aortic, paratracheal, hilar, interlobar, pulmonary ligament

缩写：SN 1, SN 2, SN1, SN2, non-SN, S L N, SLN, level I / level 1

### 3.7 解剖

子宫附件：uterus, uterine, cervix, ovary/ovaries/ovarian, fallopian tube(s), right/left tube, parametrium, myometrium, endometrium, endometrial cavity, uterine serosal surface, cornu, fundus, fundus to os, os, posterior wall

乳腺：breast, nipple, axilla, axillary

甲状腺：thyroid, hemithyroid, background thyroid

肺：lung, lobe, left lower/upper lobe, lung nodule, pleura, pleural surface, subpleural, visceral pleura

消化：gallbladder, cystic duct, liver, stomach, gastric, colon, colonic, sigmoid, ascending colon, appendix, omentum

其他：kidney, renal, skin, ear, oral cavity, maxillary sinus, vallecular, soft tissue, fibrofatty, fibroadipose

方位：right, left, bilateral, medial, lateral, superior, inferior, anterior / anteriour, posterior, superficial, deep, proximal, distal, apical, longitudinally, antero-posteriorly / anteroposteriorly

### 3.8 病变

fibroid(s), uterine fibroids, intramural, subserosal fibroids, bulging fibroids, myometrial mass, polyp, polypoid, hyperplastic polyp, fundus polyp, fibroid polyp, cyst, paratubal cyst, ovarian cyst, haemorrhagic cyst, nodule, mass, tumour, tubal mass, gastritis, intestinal metaplasia, helicobacter, dysplasia, malignancy, abscess, bulla

### 3.9 大体描述

颜色：tan, tan-coloured, whitish, pink, pinkish, red, reddish, brown, brownish, yellow, yellowish, black, blackish, greyish, blue

质地：firm, soft, solid, cystic, homogeneous, haemorrhagic, congested, indurated, irregular, oval, elongated, nodular, polypoid, smooth, puckered, translucent, membraneous, fibrofatty, fatty, unremarkable

套话：

- grossly unremarkable
- pink firm tissue on cut surfaces
- no necrosis or haemorrhage can be identified
- tan cut surfaces / homogeneous tan
- mild chronic inflammation
- no dysplasia or malignancy is seen
- no helicobacter seen, full stop
- compatible with hyperplastic polyp
- no gross tumour
- type gastric mucosa / antral type / body type

### 3.10 蜡块口令

block A–E, blocks A to C, frozen block, T-code, M-code, one/two/total N blocks, Total N 個 blocks

---

## 4. 数字、单位、中英夹杂（合成标签要混用）

真实 train 用 `language None`，同一句话里经常混：

| 现象 | 真实写法 | 合成建议 |
|---|---|---|
| 乘号 | `×`、`乘`、`x` | 三种都要；不要用 `times`/`by`（和评测 `x` 对不上） |
| 单位 | `mm`/`cm` 以及 `m m` / `c m` | 两种空格习惯都要 |
| 数字 | `5`、`五`、`two`、`兩`、`二`、`一點五` | 中文数字、英文数词、`兩` 都要出现 |
| 重量 | `gram`、`gramme` | 两种拼写 |
| 量词 | `4 個 blocks`、`四個 blocks` | 保留 `個` |
| 旁白 | `總數`、`呀`、`喎`、`加` | 少量，不要每句都插 |
| 标本序号 | `第一隻`、`第二隻`、`specimen 一` | 保留 |
| 编码 | `T-code`、`M-code` | 保留 |

测试 26SS 标本号是粘连的 `26SS11678`；模型常读成 `二六 S S 一一六七八`。合成时同一 ID 做多种读法，评测侧再归一化。

---

## 5. 建议合成配比

不要按词频均匀采样。train 已经很多 gallbladder / hyperplastic polyp / all embedded / full stop。建议：

| 桶 | 比例 | 内容 |
|---|---|---|
| A 测试缺口 | 40% | §1 全部：scrape cytology、SN/non-SN、hemithyroid、lumpectomy、submitted fresh、切缘染色、26SS 式标本号 |
| B 框架套话 | 25% | §2 模板 + 测量句 + period/full stop |
| C train 已有但要多样 | 25% | fibroids、fallopian tube、gallbladder、lymph node 全称、inked、frozen block |
| D 中英夹杂 | 10% | `乘`、`個 blocks`、`總數`、中文数字，密度接近真实 train，不要变成中文语料 |

每条合成文本尽量同时包含：**标本名 + 测量 + 一块取材动作**。只堆单词表（`pleural`、`inked`）对 ASR 帮助很小。

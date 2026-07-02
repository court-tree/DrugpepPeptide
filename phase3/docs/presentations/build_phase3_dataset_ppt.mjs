import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "E:/pep/phase3/PepCLIP_Phase3_dataset_construction_plan.pptx";
const QA_DIR = "C:/Users/admin/AppData/Local/Temp/codex-presentations/pep-phase3-dataset-ppt/qa";
const W = 1280;
const H = 720;

const C = {
  ink: "#111827",
  muted: "#4B5563",
  faint: "#F3F4F6",
  panel: "#EEF2F7",
  line: "#CBD5E1",
  blue: "#2563EB",
  teal: "#0F766E",
  amber: "#B45309",
  red: "#B91C1C",
  green: "#15803D",
  white: "#FFFFFF",
};

const FONT = "Microsoft YaHei";
const page = { left: 64, top: 48, width: 1152, height: 604 };

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: opts.size ?? 24,
    bold: opts.bold ?? false,
    color: opts.color ?? C.ink,
    fontFamily: FONT,
  };
  return shape;
}

function addBox(slide, x, y, w, h, fill = C.panel, line = C.line, radius = "rounded-md") {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: line, width: 1 },
    borderRadius: radius,
  });
}

function addRule(slide, x, y, w, color = C.line) {
  slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: w, height: 2 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
}

function addTitle(slide, title, index, kicker = "PepCLIP Phase-3") {
  addText(slide, kicker, page.left, 30, 420, 26, { size: 16, bold: true, color: C.blue });
  addText(slide, title, page.left, 62, 960, 58, { size: 44, bold: true });
  addRule(slide, page.left, 128, page.width, C.line);
  if (index) {
    addText(slide, String(index).padStart(2, "0"), 1180, 48, 44, 28, {
      size: 18,
      bold: true,
      color: C.muted,
    });
  }
}

function bulletText(items, indent = "") {
  return items.map((item) => `${indent}• ${item}`).join("\n");
}

function addBullets(slide, items, x, y, w, h, opts = {}) {
  return addText(slide, bulletText(items), x, y, w, h, {
    size: opts.size ?? 23,
    color: opts.color ?? C.ink,
  });
}

function addCode(slide, text, x, y, w, h, opts = {}) {
  addBox(slide, x, y, w, h, opts.fill ?? "#F8FAFC", opts.line ?? C.line, "rounded-sm");
  return addText(slide, text, x + 18, y + 16, w - 36, h - 28, {
    size: opts.size ?? 21,
    color: opts.color ?? C.ink,
    bold: opts.bold ?? false,
  });
}

function addLabeledBox(slide, label, body, x, y, w, h, color = C.blue) {
  addBox(slide, x, y, w, h, C.white, C.line, "rounded-sm");
  slide.shapes.add({
    geometry: "rect",
    position: { left: x, top: y, width: 7, height: h },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
  });
  addText(slide, label, x + 20, y + 16, w - 38, 32, { size: 22, bold: true, color });
  addText(slide, body, x + 20, y + 54, w - 38, h - 66, { size: 20, color: C.ink });
}

function addMiniTable(slide, headers, rows, x, y, w, rowH, colWs, opts = {}) {
  const headH = opts.headH ?? rowH;
  addBox(slide, x, y, w, headH, C.ink, C.ink, "rounded-sm");
  let cx = x;
  headers.forEach((h, i) => {
    addText(slide, h, cx + 10, y + 9, colWs[i] - 20, headH - 12, {
      size: opts.headerSize ?? 18,
      bold: true,
      color: C.white,
    });
    cx += colWs[i];
  });
  rows.forEach((row, r) => {
    const yy = y + headH + r * rowH;
    slide.shapes.add({
      geometry: "rect",
      position: { left: x, top: yy, width: w, height: rowH },
      fill: r % 2 === 0 ? C.white : "#F8FAFC",
      line: { style: "solid", fill: C.line, width: 1 },
    });
    let xx = x;
    row.forEach((cell, c) => {
      addText(slide, cell, xx + 10, yy + 8, colWs[c] - 20, rowH - 12, {
        size: opts.bodySize ?? 17,
        color: C.ink,
      });
      xx += colWs[c];
    });
  });
}

function addFlow(slide, steps, x, y, w, h, cols = 4) {
  const gap = 16;
  const rows = Math.ceil(steps.length / cols);
  const boxW = (w - gap * (cols - 1)) / cols;
  const boxH = (h - gap * (rows - 1)) / rows;
  steps.forEach((step, i) => {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const xx = x + col * (boxW + gap);
    const yy = y + row * (boxH + gap);
    addBox(slide, xx, yy, boxW, boxH, i % 2 ? "#F8FAFC" : C.panel, C.line, "rounded-sm");
    addText(slide, String(i + 1).padStart(2, "0"), xx + 14, yy + 12, 44, 26, {
      size: 18,
      bold: true,
      color: C.blue,
    });
    addText(slide, step, xx + 56, yy + 12, boxW - 70, boxH - 20, {
      size: steps.length > 8 ? 18 : 20,
      bold: true,
      color: C.ink,
    });
  });
}

function addSlide(pres, title, index, bodyFn) {
  const slide = pres.slides.add();
  slide.background.fill = C.white;
  addTitle(slide, title, index);
  bodyFn(slide);
}

async function main() {
  await fs.mkdir(path.dirname(OUT), { recursive: true });
  await fs.mkdir(QA_DIR, { recursive: true });

  const pres = Presentation.create({ slideSize: { width: W, height: H } });

  {
    const slide = pres.slides.add();
    slide.background.fill = C.white;
    addText(slide, "PepCLIP Phase-3", 72, 62, 500, 34, { size: 20, bold: true, color: C.blue });
    addText(slide, "数据集构建算法", 72, 150, 780, 78, { size: 62, bold: true });
    addText(slide, "从真实受体-肽复合物到构象增强微调数据集", 72, 246, 760, 40, {
      size: 27,
      color: C.muted,
    });
    addRule(slide, 72, 330, 560, C.blue);
    addText(slide, "V1 基线 + V1.5/V2/V3 扩展路线", 72, 360, 710, 32, {
      size: 23,
      color: C.ink,
    });
    addBox(slide, 850, 96, 300, 424, "#F8FAFC", C.line, "rounded-sm");
    addText(slide, "本 PPT 回答", 884, 130, 230, 34, { size: 26, bold: true });
    addBullets(
      slide,
      [
        "第一版数据集怎么构建",
        "每一步输入、处理、输出是什么",
        "后续构象增强如何接到同一条流水线",
      ],
      884,
      190,
      230,
      230,
      { size: 22 },
    );
  }

  addSlide(pres, "Phase-3 的目标：真实结构监督微调", 2, (slide) => {
    addLabeledBox(slide, "Phase-2 已完成", "教师数据 -> 双塔预训练 -> 配对检索验证 -> 原始 Top-K 检索", 72, 168, 510, 130, C.teal);
    addLabeledBox(slide, "Phase-3 要补上的监督", "受体界面 <-> 真实肽结合构象", 72, 332, 510, 130, C.blue);
    addBox(slide, 666, 168, 490, 294, "#F8FAFC", C.line, "rounded-sm");
    addText(slide, "核心目标", 696, 196, 240, 34, { size: 28, bold: true });
    addText(slide, "让模型不仅知道哪条肽可能结合当前受体界面，还要知道这条肽的哪种结合构象更适合当前界面。", 696, 252, 420, 142, { size: 26, color: C.ink });
  });

  addSlide(pres, "总体路线：先构建可信 V1，再逐步增强构象监督", 3, (slide) => {
    addFlow(
      slide,
      [
        "V1\n真实结合配对\n只用强正样本",
        "V1.5\n完全同序列构象证据\n证据统计和审计",
        "V2\n同簇弱正样本\n构象困难负样本排序",
        "V3\n相似序列 / 模体先验\n只作为先验",
      ],
      72,
      172,
      1136,
      170,
      4,
    );
    addText(slide, "为什么要分版本", 72, 396, 320, 34, { size: 30, bold: true });
    addMiniTable(
      slide,
      ["版本", "回答的问题"],
      [
        ["V1", "真实结合构象微调有没有用？"],
        ["V1.5", "同序列构象证据够不够支撑增强？"],
        ["V2", "构象增强训练是否提升受体-构象匹配？"],
        ["V3", "相似序列 / 模体先验是否提供额外信息？"],
      ],
      72,
      448,
      1010,
      44,
      [120, 890],
      { bodySize: 18 },
    );
  });

  addSlide(pres, "V1 是 Phase-3 的最小可用数据集", 4, (slide) => {
    addCode(slide, "锚点样本 = 受体界面 + 真实肽结合构象", 88, 178, 720, 76, { size: 29, bold: true });
    addLabeledBox(slide, "V1 只生成一种训练监督边", "强正样本：\n受体界面 <-> 真实肽结合构象", 88, 290, 520, 150, C.blue);
    addLabeledBox(slide, "V1 的目的", "先建立一个干净问题：真实复合物里的受体界面和肽结合构象，能否作为 Phase-3 微调的有效强监督？", 660, 290, 480, 150, C.teal);
    addText(slide, "V1 不做", 88, 492, 160, 30, { size: 27, bold: true });
    addText(slide, "同簇弱正样本 / 构象困难负样本 / 相似序列先验 / 模体先验 / 多正样本损失 / 排序损失", 248, 493, 880, 64, { size: 22, color: C.muted });
  });

  addSlide(pres, "第一步：收集真实受体-肽复合物候选", 5, (slide) => {
    addMiniTable(
      slide,
      ["优先数据源", "原因"],
      [
        ["Q-BioLiP / BioLiP", "已有受体-肽复合物注释"],
        ["PepBDB / Propedia", "补充真实肽结合结构"],
        ["PDB 蛋白-肽记录", "作为结构级候选来源"],
      ],
      72,
      168,
      620,
      58,
      [260, 360],
      { bodySize: 18 },
    );
    addLabeledBox(slide, "每条候选记录需要包含", "来源数据库、来源编号、PDB 编号、生物组装编号、结构文件、受体链、肽链、残基范围、来源可信度。", 730, 168, 430, 210, C.blue);
    addLabeledBox(slide, "不优先使用", "只有肽序列、只有活性信息、没有明确受体结合结构、没有受体链-肽链对应关系的数据。", 730, 416, 430, 150, C.red);
  });

  addSlide(pres, "V1 数据集构建流水线", 6, (slide) => {
    addFlow(
      slide,
      [
        "收集真实复合物候选",
        "多数据库候选去重",
        "重建生物组装结构",
        "定位受体链 / 肽链",
        "过滤肽链",
        "验证受体-肽接触",
        "定义受体界面",
        "提取真实结合构象",
        "质量检查与锚点生成",
        "防泄露划分",
        "导出训练数据 A / B",
        "生成数据集审计",
      ],
      72,
      164,
      1136,
      420,
      4,
    );
  });

  addSlide(pres, "V1 步骤 1-2：候选收集与去重", 7, (slide) => {
    addLabeledBox(slide, "步骤 1：候选收集", "每条记录必须能定位到 PDB 编号、生物组装编号、受体链、肽链、肽序列 / 残基范围和结构文件。", 72, 170, 520, 178, C.blue);
    addLabeledBox(slide, "步骤 2：多数据库去重", "最低去重键：PDB 编号 + 生物组装编号 + 受体链 + 肽链。同一个复合物只保留一条主记录，同时保存来源列表。", 72, 388, 520, 178, C.teal);
    addBox(slide, 670, 176, 420, 280, "#F8FAFC", C.line, "rounded-sm");
    addText(slide, "目的", 704, 210, 220, 34, { size: 30, bold: true });
    addBullets(slide, ["避免同一个锚点样本重复进入训练", "避免同一个真实复合物跨数据划分泄露", "保留多来源证据，便于后续审计"], 704, 270, 330, 142, { size: 23 });
  });

  addSlide(pres, "V1 步骤 3-5：结构重建与肽链过滤", 8, (slide) => {
    addLabeledBox(slide, "步骤 3：生物组装结构", "读取 PDB/mmCIF -> 使用生物组装结构 -> 定位受体链和肽链。避免非对称单元带来的晶体接触误判。", 72, 166, 520, 176, C.blue);
    addLabeledBox(slide, "步骤 4：肽链过滤", "只保留 8 <= 肽长度 <= 20；过滤残基不连续、主链缺失、非标准残基过多、缺失过多、非独立肽链等情况。", 72, 382, 520, 176, C.teal);
    addMiniTable(
      slide,
      ["长度", "长度分组"],
      [
        ["8-10 aa", "短肽组"],
        ["11-15 aa", "中短肽组"],
        ["16-20 aa", "中长肽组"],
      ],
      684,
      212,
      420,
      54,
      [150, 270],
      { bodySize: 18 },
    );
  });

  addSlide(pres, "V1 步骤 6-7：接触验证与界面定义", 9, (slide) => {
    addCode(slide, "最小重原子距离 <= 5.0 A", 88, 168, 520, 72, { size: 28, bold: true });
    addText(slide, "并按肽长度检查接触数量与界面残基数量。短肽接触数自然较少但每个残基更关键，中长肽需要更高接触数量保证真实界面。", 88, 268, 520, 116, { size: 23, color: C.ink });
    addLabeledBox(slide, "5A 核心界面", "受体中距离肽重原子 <= 5 A 的残基；用于质量控制和核心接触统计。", 680, 168, 430, 130, C.blue);
    addLabeledBox(slide, "10A 上下文区域", "受体中距离肽重原子 <= 10 A 的残基；用于模型输入，提供周围结构环境。", 680, 336, 430, 130, C.teal);
  });

  addSlide(pres, "V1 步骤 8-9：真实结合构象与锚点样本", 10, (slide) => {
    addLabeledBox(slide, "提取真实结合构象", "从真实复合物肽链中提取肽序列、主链坐标 N/CA/C/O、全部非氢重原子坐标。", 72, 166, 520, 168, C.blue);
    addLabeledBox(slide, "质量检查失败则删除整个锚点", "主链原子缺失、残基不连续、序列对不上、缺失比例超标或坐标质量太差时，不保留该锚点样本。", 72, 374, 520, 168, C.red);
    addCode(slide, "锚点样本 = 受体界面 + 真实结合构象", 670, 204, 470, 86, { size: 27, bold: true });
    addText(slide, "不能删除真实结合构象后继续保留受体界面。V1 的强正样本标签就来自这个真实结合构象。", 690, 334, 410, 118, { size: 24, color: C.ink });
  });

  addSlide(pres, "V1 步骤 10：数据划分与防泄露", 11, (slide) => {
    addText(slide, "为什么不能随机切配对", 72, 168, 380, 34, { size: 29, bold: true });
    addBullets(slide, ["同一肽序列可能同时出现在训练集和测试集", "同一受体家族可能同时出现在训练集和测试集", "同一 PDB 或高度相似结构可能跨划分泄露"], 72, 224, 520, 150, { size: 23 });
    addMiniTable(
      slide,
      ["推荐划分层级", "作用"],
      [
        ["配对级划分", "基础合理性检查"],
        ["肽精确序列划分", "避免同序列泄露"],
        ["受体家族划分", "避免受体家族泄露"],
        ["严格划分", "同时控制肽和受体相似性"],
      ],
      650,
      168,
      500,
      52,
      [210, 290],
      { bodySize: 17 },
    );
    addCode(slide, "每个样本保存：划分分组 / 所属集合 / 肽序列键 / 受体家族键 / PDB 键", 72, 470, 1000, 76, { size: 22 });
  });

  addSlide(pres, "V1 输出表与训练入口", 12, (slide) => {
    addMiniTable(
      slide,
      ["输出", "作用"],
      [
        ["receptor_peptide_pair.jsonl", "锚点 / 受体 / 肽配对信息"],
        ["peptide_conformer_evidence.jsonl", "真实结合构象证据"],
        ["conformer_cluster.jsonl", "结合构象的初始聚类 / 相似性信息"],
        ["track_a_<split>.jsonl", "序列侧 / 受体侧训练输入"],
        ["track_b_<split>.jsonl", "构象侧 / 三维训练输入"],
        ["dataset_audit.json", "数量、过滤、划分统计、泄露检查"],
      ],
      72,
      164,
      760,
      48,
      [330, 430],
      { bodySize: 16 },
    );
    addBox(slide, 868, 198, 324, 184, C.white, C.line, "rounded-sm");
    slide.shapes.add({
      geometry: "rect",
      position: { left: 868, top: 198, width: 7, height: 184 },
      fill: C.blue,
      line: { style: "solid", fill: C.blue, width: 0 },
    });
    addText(slide, "V1 训练与评估", 896, 224, 250, 30, { size: 24, bold: true, color: C.blue });
    addText(slide, "训练集：\n只用强正样本\n\n验证/测试集：\n只用强正样本", 896, 268, 260, 100, { size: 17 });
    addBox(slide, 868, 416, 324, 124, "#F8FAFC", C.line, "rounded-sm");
    addText(slide, "V1 的成果是一套可以训练、可以评估、可以审计的真实结构强监督数据集。", 896, 438, 260, 76, { size: 19, color: C.ink });
  });

  addSlide(pres, "V1.5：完全同序列构象证据层", 13, (slide) => {
    addLabeledBox(slide, "输入", "V1 锚点样本\nV1 真实结合构象\nPDB 序列与 mmCIF 精确匹配证据", 72, 168, 330, 150, C.blue);
    addLabeledBox(slide, "处理", "对每个锚点肽序列搜索完全同序列构象：序列一致、覆盖完整、长度相同、无缺口。", 450, 168, 370, 150, C.teal);
    addLabeledBox(slide, "硬规则", "同序列构象池 = 真实结合构象 + 外部完全同序列构象", 868, 168, 300, 150, C.amber);
    addCode(slide, "输出：外部精确匹配构象 / 肽构象证据 / 构象聚类 / 真实结合构象到聚类的映射 / 构象挖掘摘要 / 数据集审计", 72, 384, 1050, 104, { size: 21 });
    addText(slide, "V1.5 默认只做覆盖率、聚类质量、真实构象映射和泄露审计，为 V2 做准备。", 72, 526, 980, 42, { size: 25, bold: true, color: C.ink });
  });

  addSlide(pres, "V2：构象增强训练怎么接入", 14, (slide) => {
    addLabeledBox(slide, "V2 输入", "V1 锚点样本\nV1.5 同序列构象池\n完全同序列构象聚类\n真实结合构象所属聚类", 72, 168, 330, 190, C.blue);
    addMiniTable(
      slide,
      ["监督边", "来源与训练语义"],
      [
        ["强正样本", "真实结合构象；权重 = 1.0"],
        ["同簇弱正样本", "同序列 + 同构象簇；只用于训练；权重 0.3-0.5"],
        ["构象困难负样本", "同序列 + 不同构象簇；只用于训练；只进排序损失"],
      ],
      450,
      168,
      700,
      68,
      [250, 450],
      { bodySize: 17 },
    );
    addText(slide, "V2 不重新定义正样本来源。真实强正样本仍来自 V1 的真实结合构象；V2 只是在其周围增加同序列构象增强。", 72, 450, 960, 76, { size: 26, bold: true, color: C.ink });
  });

  addSlide(pres, "V2 的训练改变在损失函数，不在 V1 强监督定义", 15, (slide) => {
    addCode(slide, "总损失 = 一维序列检索损失 + 三维多正样本损失 + 融合多正样本损失 + 构象排序损失", 72, 162, 1060, 76, { size: 24, bold: true });
    addMiniTable(
      slide,
      ["分支", "使用样本", "损失"],
      [
        ["一维序列分支", "只用强正样本", "普通对比学习"],
        ["三维构象分支", "强正样本 + 同簇弱正样本", "多正样本对比学习"],
        ["融合分支", "强正样本 + 同簇弱正样本", "多正样本对比学习"],
        ["排序分支", "真实结合构象 vs 同序列异簇构象", "间隔排序损失"],
      ],
      72,
      286,
      840,
      52,
      [190, 410, 240],
      { bodySize: 16 },
    );
    addBox(slide, 930, 276, 280, 260, C.white, C.line, "rounded-sm");
    slide.shapes.add({
      geometry: "rect",
      position: { left: 930, top: 276, width: 7, height: 260 },
      fill: C.red,
      line: { style: "solid", fill: C.red, width: 0 },
    });
    addText(slide, "语义边界", 960, 302, 210, 30, { size: 24, bold: true, color: C.red });
    addBullets(
      slide,
      [
        "不是肽序列级负样本",
        "不进一维序列分支",
        "不进普通负样本池",
        "不进多正样本分母",
      ],
      960,
      350,
      220,
      166,
      { size: 16 },
    );
  });

  addSlide(pres, "V3：相似序列 / 模体先验怎么接入", 16, (slide) => {
    addLabeledBox(slide, "V3 动机", "当完全同序列构象较少时，可以搜索相似序列构象和模体家族构象。", 72, 166, 390, 150, C.blue);
    addLabeledBox(slide, "它代表什么", "相似肽 / 模体家族的构象先验；不是当前肽序列的真实构象监督。", 72, 356, 390, 150, C.teal);
    addCode(slide, "构象先验表\n\n锚点编号 / 构象编号 / 先验类型 / 序列一致性 / 序列相似性 / 覆盖度 / 接触核心一致性 / 接触核心相似性 / 原因", 540, 166, 560, 190, { size: 20 });
    addLabeledBox(slide, "硬边界", "V3 先验不参与同序列构象聚类、监督边生成、同簇弱正样本或构象困难负样本。", 540, 402, 560, 124, C.red);
  });

  addSlide(pres, "无论哪个版本，主评估都必须保持清洁", 17, (slide) => {
    addLabeledBox(slide, "主评估只使用", "强正样本", 72, 168, 330, 116, C.blue);
    addLabeledBox(slide, "主评估回答", "模型能否识别真实受体-肽结合配对？", 72, 330, 330, 116, C.teal);
    addLabeledBox(slide, "不进入主评估", "同簇弱正样本\n构象困难负样本\n相似序列构象先验\n模体构象先验", 470, 168, 340, 200, C.red);
    addLabeledBox(slide, "可选诊断", "构象排序诊断：比较真实结合构象和同序列异簇构象的分数。\n\n先验诊断：分析先验是否提供额外排序信息。", 860, 168, 300, 250, C.amber);
    addText(slide, "主指标衡量真实配对检索；诊断指标衡量构象增强策略。两者必须分开汇报。", 72, 522, 920, 38, { size: 26, bold: true });
  });

  addSlide(pres, "阶段三数据集构建的实际实施顺序", 18, (slide) => {
    addFlow(
      slide,
      [
        "完成 V1 数据集\n只用强正样本",
        "用 V1 跑基线训练\n得到第一组结果",
        "构建 V1.5 精确构象证据\n覆盖率 / 聚类 / 泄露审计",
        "实现 V2 监督边\n同簇弱正 + 困难负样本",
        "V2 稳定后实现 V3 先验\n消融 / 重排序",
      ],
      72,
      164,
      960,
      250,
      3,
    );
    addMiniTable(
      slide,
      ["阶段", "交付物"],
      [
        ["V1", "基线数据集 + 数据审计 + 基线训练结果"],
        ["V1.5", "构象证据报告 + 聚类映射 + 审计"],
        ["V2", "监督边 + 多正样本 / 排序训练消融"],
        ["V3", "构象先验 + 先验消融 / 重排序分析"],
      ],
      72,
      464,
      860,
      44,
      [140, 720],
      { bodySize: 17 },
    );
  });

  for (const [index, slide] of pres.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await pres.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(QA_DIR, `${stem}.png`), Buffer.from(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }

  const montage = await pres.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(QA_DIR, "deck-montage.webp"), Buffer.from(await montage.arrayBuffer()));

  const pptx = await PresentationFile.exportPptx(pres);
  await pptx.save(OUT);
  console.log(OUT);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

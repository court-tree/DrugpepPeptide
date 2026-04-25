const fs = await import("node:fs/promises");
const path = await import("node:path");
const { Presentation, PresentationFile } = await import("@oai/artifact-tool");

const W = 1280;
const H = 720;
const OUT_DIR = "E:\\pep\\phase1\\ppt_adaptive_coverage";
const SCRATCH_DIR = path.join(OUT_DIR, "tmp");
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");

const C = {
  navy: "#102A43",
  ink: "#15202B",
  muted: "#657786",
  paper: "#F7F3EA",
  card: "#FFFFFF",
  line: "#D8DEE6",
  teal: "#12B886",
  tealDark: "#087F5B",
  amber: "#F59F00",
  coral: "#E8590C",
  blue: "#3B82F6",
  lavender: "#7048E8",
};

const FONT_TITLE = "Microsoft YaHei";
const FONT_BODY = "Microsoft YaHei";
const TOTAL = 12;

function addRect(slide, left, top, width, height, fill, line = null, radius = 0) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: line || { fill: "#00000000", width: 0 },
  });
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const s = slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill: "#00000000",
    line: { fill: "#00000000", width: 0 },
  });
  s.text = text;
  s.text.typeface = opts.face || FONT_BODY;
  s.text.fontSize = opts.size || 24;
  s.text.color = opts.color || C.ink;
  s.text.bold = opts.bold || false;
  s.text.alignment = opts.align || "left";
  s.text.verticalAlignment = opts.valign || "top";
  s.text.autoFit = "shrinkText";
  s.text.insets = opts.insets || { left: 8, right: 8, top: 4, bottom: 4 };
  return s;
}

function title(slide, kicker, main, sub = "") {
  addText(slide, kicker, 58, 34, 620, 30, { size: 17, color: C.tealDark, bold: true });
  addText(slide, main, 54, 68, 980, 76, { size: 33, color: C.navy, bold: true, face: FONT_TITLE });
  if (sub) addText(slide, sub, 58, 145, 1000, 52, { size: 19, color: C.muted });
}

function footer(slide, n) {
  addText(slide, `Phase-1 逻辑与调参复盘 · ${n}/${TOTAL}`, 910, 666, 310, 26, {
    size: 13,
    color: C.muted,
    align: "right",
  });
}

function addMetricCard(slide, x, y, w, h, value, label, note, color = C.teal) {
  addRect(slide, x, y, w, h, C.card, { fill: C.line, width: 1 }, 1);
  addRect(slide, x, y, 8, h, color);
  addText(slide, value, x + 24, y + 18, w - 40, 42, { size: 30, bold: true, color });
  addText(slide, label, x + 24, y + 66, w - 40, 46, { size: 18, bold: true, color: C.ink });
  addText(slide, note, x + 24, y + 116, w - 40, h - 128, { size: 15, color: C.muted });
}

function addCard(slide, x, y, w, h, head, body, color = C.teal) {
  addRect(slide, x, y, w, h, C.card, { fill: C.line, width: 1 }, 1);
  addRect(slide, x, y, w, 9, color);
  addText(slide, head, x + 20, y + 24, w - 40, 34, { size: 22, bold: true, color: C.navy });
  addText(slide, body, x + 20, y + 72, w - 40, h - 84, { size: 16, color: C.ink });
}

function addSimpleTable(slide, rows, x, y, w, h, headerFill = C.navy) {
  const rowH = h / rows.length;
  const colW = w / rows[0].length;
  addRect(slide, x, y, w, h, C.card, { fill: C.line, width: 1 }, 1);
  for (let r = 0; r < rows.length; r += 1) {
    for (let c = 0; c < rows[0].length; c += 1) {
      const left = x + c * colW;
      const top = y + r * rowH;
      const fill = r === 0 ? headerFill : (r % 2 === 0 ? "#F8FAFC" : "#FFFFFF");
      addRect(slide, left, top, colW, rowH, fill, { fill: C.line, width: 0.7 });
      addText(slide, rows[r][c], left + 6, top + 5, colW - 12, rowH - 8, {
        size: r === 0 ? 14 : 13,
        bold: r === 0,
        color: r === 0 ? "#FFFFFF" : C.ink,
        align: c === 0 ? "left" : "center",
        valign: "middle",
        insets: { left: 4, right: 4, top: 2, bottom: 2 },
      });
    }
  }
}

function createDeck() {
  const p = Presentation.create({ slideSize: { width: W, height: H } });
  p.theme.colorScheme = {
    name: "PepCLIP Phase1",
    themeColors: { accent1: C.teal, accent2: C.amber, bg1: C.paper, tx1: C.ink },
  };

  // 1
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    addRect(slide, 0, 0, 1280, 720, C.paper);
    addRect(slide, 746, -80, 620, 880, C.navy);
    addRect(slide, 800, 70, 370, 510, "#1D4E68", null, 1);
    addText(slide, "Phase-1", 870, 138, 230, 40, { size: 34, color: "#FFFFFF", bold: true, align: "center" });
    addText(slide, "算法逻辑与\n最新全量结果复盘", 58, 105, 640, 160, {
      size: 42, color: C.navy, bold: true, face: FONT_TITLE,
    });
    addText(slide, "当前主线：热点锚点生成 8-20 aa 候选，单条质量过滤，平均接触数带权抽样，受体+多肽联合去重。", 64, 292, 650, 82, {
      size: 23, color: C.ink, bold: true,
    });
    addText(slide, "最新结论：Step3 按长度段保留候选有效，full_run_v4 成为当前推荐主结果。", 64, 396, 650, 56, {
      size: 22, color: C.coral, bold: true,
    });
    addMetricCard(slide, 64, 500, 190, 115, "7", "Phase-1 步骤", "从质控到 metadata", C.teal);
    addMetricCard(slide, 280, 500, 190, 115, "170k", "full_run_v4", "最终样本量", C.blue);
    addMetricCard(slide, 496, 500, 190, 115, "10.65", "平均长度", "较 v2 明显提升", C.lavender);
    footer(slide, 1);
  }

  // 2
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "ALGORITHM OVERVIEW", "当前 Phase-1 主线", "先找少数真实界面热点，再生成连续窗口；先保证单条质量，再做代表性抽样和联合去重。");
    const steps = [
      ["1", "结构质控", "只保留结构和界面可靠的复合物"],
      ["2", "双向任务", "链对拆成 receptor / peptide-source"],
      ["3", "热点候选", "少数 anchor 生成 8-20 aa 窗口"],
      ["4", "单条过滤", "连续性 + 接触质量"],
      ["5", "带权抽样", "avg_contact_count + 8-mer 上限"],
      ["6", "联合去重", "receptor + peptide 同时高同源才删"],
      ["7", "metadata", "生成训练所需序列与 patch 信息"],
    ];
    let x = 50;
    for (const [num, head, body] of steps) {
      addRect(slide, x, 255, 156, 210, C.card, { fill: C.line, width: 1 }, 1);
      addRect(slide, x + 18, 232, 50, 50, C.teal, null, 1);
      addText(slide, num, x + 18, 240, 50, 30, { size: 24, color: "#FFFFFF", bold: true, align: "center" });
      addText(slide, head, x + 16, 305, 124, 36, { size: 22, bold: true, color: C.navy, align: "center" });
      addText(slide, body, x + 16, 356, 124, 74, { size: 15, color: C.ink, align: "center" });
      x += 170;
    }
    addRect(slide, 105, 548, 1060, 56, "#E7F5FF", { fill: "#A5D8FF", width: 1 }, 1);
    addText(slide, "一句话：热点锚点生成候选，单条质量过滤，平均接触数抽样，联合受体/多肽去重。", 135, 562, 1000, 28, {
      size: 22, bold: true, color: C.navy, align: "center",
    });
    footer(slide, 2);
  }

  // 3
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "STEPS 1-3", "从结构质控到热点候选生成", "核心变化在 Step3：不再纯 top16，而是按长度段给不同候选留入口。");
    addCard(slide, 70, 235, 330, 260, "Step 1 结构级质控", "读取 mmCIF；保留有效蛋白链数量足够的结构；检查链间有效接触和最小界面面积；过滤蛋白链不足、无接触、弱界面结构。", C.blue);
    addCard(slide, 475, 235, 330, 260, "Step 2 双向任务生成", "对每对接触链生成两个方向：A 作 receptor、B 作 peptide source；B 作 receptor、A 作 peptide source。", C.teal);
    addCard(slide, 880, 235, 330, 260, "Step 3 热点锚点窗口", "seed residue 需直接接触 receptor；anchor 按直接接触数和局部 avg_contact_count 排序；NMS 去相邻重复；每 task 最多 3 个 anchor。", C.amber);
    addText(slide, "Step3 默认：min_anchor_contact_count=2；max_anchors_per_task=3；max_candidates_per_task=16；窗口长度 8-20 aa。", 92, 535, 1040, 34, {
      size: 20, bold: true, color: C.navy, align: "center",
    });
    addText(slide, "最新已实现：8-10 aa 最多 6 条，11-14 aa 最多 6 条，15-20 aa 最多 4 条；不足时按 avg_contact_count 回填。", 92, 580, 1040, 34, {
      size: 20, bold: true, color: C.coral, align: "center",
    });
    footer(slide, 3);
  }

  // 4
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "STEPS 4-5", "单条质量过滤 + 带权抽样", "Step4 不做近重复去冗余，只判断单条窗口是否结构连续、接触充分；Step5 再保留代表样本。");
    addCard(slide, 85, 235, 500, 280, "Step 4 单条候选质量过滤", "回原始结构重新切出窗口；检查窗口边界和主链连续性；重新计算 avg_contact_count 与 contact_coverage；当前保守基线条件为 avg_contact_count >= 3.5 且 contact_coverage >= 0.5。", C.coral);
    addCard(slide, 690, 235, 500, 280, "Step 5 带权抽样与长度约束", "每个 task 最多保留 4 条；sampling_weight = avg_contact_count；加入 max_len8_per_task=2：如果存在非 8-mer，则最多保留 2 条 8-mer；如果只有 8-mer，则允许兜底。", C.teal);
    addRect(slide, 160, 555, 960, 58, "#FFF4E6", { fill: "#FFD8A8", width: 1 }, 1);
    addText(slide, "目标：先让进入抽样池的候选“说得过去”，再用随机性和轻量长度约束避免 task 内完全被短肽占满。", 190, 570, 900, 28, {
      size: 21, bold: true, color: C.navy, align: "center",
    });
    footer(slide, 4);
  }

  // 5
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "STEPS 6-7", "联合 receptor + peptide 去重，再生成 metadata", "只删除真正近重复样本，不因为 receptor 相似就误删不同 peptide。");
    addCard(slide, 80, 225, 520, 300, "Step 6 联合同源去重", "给每条候选补 receptor sequence 和 peptide sequence；代表优先级：长肽优先、avg_contact_count 高优先、coverage 高优先；只有 receptor_identity >= 0.85、peptide_identity >= 0.85 且短/长长度覆盖 >= 0.70 时才视为重复。", C.lavender);
    addCard(slide, 690, 225, 500, 300, "Step 7 最终 metadata", "读取 main 和 monitor；回原始结构定位 peptide window；生成 peptide sequence、receptor local patch residue ids、peptide residue ids、split 标记和 proxy cap 信息；patch_cutoff = 6.0。", C.blue);
    addText(slide, "关键原则：同一个 receptor pocket 可以结合不同 peptide；必须 receptor 和 peptide 同时高度相似，才删除。", 130, 565, 1020, 34, {
      size: 22, bold: true, color: C.coral, align: "center",
    });
    footer(slide, 5);
  }

  // 6
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "DEFAULT PARAMETERS", "当前核心默认参数", "当前推荐主结果以 full_run_v4 对应参数为准。");
    addSimpleTable(slide, [
      ["Step", "参数", "当前值", "作用"],
      ["Step 3", "min_anchor_contact_count", "2", "定义热点 seed 的最低直接接触"],
      ["Step 3", "max_anchors_per_task", "3", "控制每个 task 的热点数量"],
      ["Step 3", "max_candidates_per_task", "16", "控制候选规模"],
      ["Step 3", "8-10 / 11-14 / 15-20", "6 / 6 / 4", "长度段候选保留"],
      ["Step 4", "min_avg_contact_count", "3.5", "保证平均接触质量"],
      ["Step 4", "min_contact_coverage", "0.5", "保守基线 coverage 阈值"],
      ["Step 5", "max_keep_per_task / max_len8_per_task", "4 / 2", "抽样规模与 8-mer 上限"],
      ["Step 6", "identity / min_coverage", "0.85 / 0.70", "联合去重条件"],
      ["Step 7", "patch_cutoff", "6.0", "局部 receptor patch 半径"],
    ], 75, 205, 1120, 430);
    footer(slide, 6);
  }

  // 7
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "VERSION MAP", "full_run_v2 / v3 / v4 到底分别改了什么", "这一页专门说明版本差异，避免把 Step4 和 Step3 的贡献混在一起。");
    addSimpleTable(slide, [
      ["版本", "Step3", "Step4", "核心目的"],
      ["full_run_v2", "旧版候选保留：更接近纯 avg_contact_count top16", "保守基线：avg_contact_count >= 3.5 且 coverage >= 0.5", "作为保守基线版本"],
      ["full_run_v3", "与 v2 相同", "长度自适应 coverage：8-10=0.50，11-14=0.40，15-20=0.30", "验证 Step4 是否是长肽不足主瓶颈"],
      ["full_run_v4", "按长度段保留候选：8-10=6，11-14=6，15-20=4", "延续当前长度自适应 Step4", "验证 Step3 长度段保留是否真正改善分布"],
    ], 70, 230, 1140, 250);
    addMetricCard(slide, 120, 535, 280, 120, "v3", "反证实验", "证明 Step4 不是主要瓶颈", C.coral);
    addMetricCard(slide, 500, 535, 280, 120, "v4", "有效改进", "证明 Step3 长度段保留真正起效", C.teal);
    addMetricCard(slide, 880, 535, 280, 120, "当前主用", "推荐版本", "full_run_v4", C.blue);
    footer(slide, 7);
  }

  // 8
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "NEGATIVE RESULT", "Step4 coverage 自适应是反证实验，不是最终解", "full_run_v3 复用 v2 的 Step1-Step3，只改 Step4 coverage；结果证明 Step4 不是主要瓶颈。");
    addRect(slide, 70, 235, 500, 275, C.card, { fill: C.line, width: 1 }, 1);
    addText(slide, "full_run_v2", 100, 260, 220, 38, { size: 30, bold: true, color: C.navy });
    addText(slide, "固定 coverage 阈值", 100, 310, 300, 32, { size: 22, bold: true });
    addText(slide, "contact_coverage >= 0.50\navg_contact_count >= 3.5", 100, 365, 390, 90, { size: 24, color: C.tealDark, bold: true });
    addRect(slide, 710, 235, 500, 275, C.card, { fill: C.line, width: 1 }, 1);
    addText(slide, "full_run_v3", 740, 260, 220, 38, { size: 30, bold: true, color: C.navy });
    addText(slide, "长度自适应 coverage", 740, 310, 330, 32, { size: 22, bold: true });
    addText(slide, "8-10 aa >= 0.50\n11-14 aa >= 0.40\n15-20 aa >= 0.30\navg_contact_count >= 3.5", 740, 356, 390, 120, { size: 22, color: C.tealDark, bold: true });
    addText(slide, "结果：15-20 aa 从 2,213 到 2,200，几乎没有改善。", 145, 565, 980, 34, { size: 24, bold: true, color: C.coral, align: "center" });
    footer(slide, 8);
  }

  // 9
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "WHY V3 FAILED", "v3 说明：长肽不足不是主要死在 Step4", "Step4 放宽 coverage 后，整体长度结构几乎不动，因此问题更可能在 Step3 或 Step5。");
    addSimpleTable(slide, [
      ["指标", "full_run_v2", "full_run_v3", "变化"],
      ["最终样本数", "171,123", "171,042", "-81"],
      ["平均 peptide 长度", "9.492", "9.490", "几乎不变"],
      ["平均 avg_contact_count", "4.928", "4.926", "几乎不变"],
      ["最小 contact_coverage", "0.5", "0.4", "阈值生效"],
      ["平均 contact_coverage", "0.9181", "0.9179", "几乎不变"],
      ["15-20 aa 总数", "2,213", "2,200", "-13"],
    ], 72, 225, 820, 360);
    addRect(slide, 940, 235, 245, 300, "#FFF4E6", { fill: "#FFD8A8", width: 1 }, 1);
    addText(slide, "解释", 968, 260, 190, 34, { size: 25, bold: true, color: C.coral });
    addText(slide, "如果 Step4 coverage 是主瓶颈，v3 应该明显增加 15-20 aa。\n\n但事实没有发生，因此主要问题不在 Step4。", 968, 315, 185, 150, { size: 19, color: C.ink });
    addText(slide, "这是一条重要反证。", 968, 486, 190, 45, { size: 20, bold: true, color: C.coral });
    footer(slide, 9);
  }

  // 10
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "EFFECTIVE FIX", "full_run_v4：Step3 长度段候选保留真正起效", "v4 带着当前 Step4 一起运行，但真正拉动长度分布的主因是 Step3 长度段候选保留。");
    addSimpleTable(slide, [
      ["指标", "full_run_v2", "full_run_v4", "变化"],
      ["最终样本数", "171,123", "170,269", "-854"],
      ["平均 peptide 长度", "9.49", "10.65", "明显上升"],
      ["平均 avg_contact_count", "4.93", "4.85", "小幅下降"],
      ["平均 contact_coverage", "0.918", "0.915", "小幅下降"],
      ["平均 longest_contact_run", "7.92", "8.61", "上升"],
      ["15-20 aa 总数", "2,213", "25,309", "+23,096"],
    ], 72, 225, 820, 360);
    addRect(slide, 940, 235, 245, 300, "#E6FCF5", { fill: "#96F2D7", width: 1 }, 1);
    addText(slide, "判断", 968, 260, 190, 34, { size: 25, bold: true, color: C.tealDark });
    addText(slide, "长肽从几乎没有，变成真正进入数据集的一部分。\n\n总样本量稳定，质量只轻微下降，说明这次改动是有效解。", 968, 315, 185, 160, { size: 19, color: C.ink });
    addText(slide, "当前推荐主结果：v4", 968, 486, 190, 45, { size: 20, bold: true, color: C.tealDark });
    footer(slide, 10);
  }

  // 11
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "LENGTH DISTRIBUTION", "v4 的长度分布明显更健康", "短肽不再垄断，中长肽显著抬升；这是 Step3 长度段候选保留最直接的结果。");
    addMetricCard(slide, 75, 240, 220, 220, "45,355", "8-mer", "较 v2 减少 10,974", C.blue);
    addMetricCard(slide, 325, 240, 220, 220, "30,143", "11-mer", "较 v2 增加 12,084", C.teal);
    addMetricCard(slide, 575, 240, 220, 220, "17,799", "12-mer", "较 v2 增加 9,816", C.amber);
    addMetricCard(slide, 825, 240, 220, 220, "15,587", "15-mer", "较 v2 增加 14,642", C.coral);
    addMetricCard(slide, 1075, 240, 140, 220, "430", "20-mer", "较 v2 增加 350", C.lavender);
    addRect(slide, 110, 530, 1040, 70, "#E7F5FF", { fill: "#A5D8FF", width: 1 }, 1);
    addText(slide, "长度统计：p50 = 10，p75 = 12，mean = 10.65；相比 v2 的 mean = 9.49，长度结构明显改善。", 140, 552, 980, 30, {
      size: 22, bold: true, color: C.navy, align: "center",
    });
    footer(slide, 11);
  }

  // 12
  {
    const slide = p.slides.add();
    slide.background.fill = C.paper;
    title(slide, "FINAL TAKEAWAY", "当前推荐版本：保留 v2，主用 v4", "v2 是保守基线，v3 是 Step4 反证实验，v4 是当前更合理的主结果。");
    addCard(slide, 80, 235, 330, 280, "保留 full_run_v2", "作为保守基线版本：结构质量强、分布偏短、适合做对照。", C.blue);
    addCard(slide, 475, 235, 330, 280, "保留 full_run_v3", "作为 Step4 coverage 反证实验：证明 Step4 不是长肽不足的主瓶颈。", C.coral);
    addCard(slide, 870, 235, 330, 280, "主用 full_run_v4", "作为当前推荐数据底座：长度分布显著改善，总量稳定，质量只轻微下降。", C.teal);
    addRect(slide, 120, 560, 1000, 58, "#FFF4E6", { fill: "#FFD8A8", width: 1 }, 1);
    addText(slide, "下一步重点不是继续调 Step4，而是围绕 Step3 / Step5 做更细的长度与质量平衡。", 150, 575, 940, 28, {
      size: 22, bold: true, color: C.navy, align: "center",
    });
    footer(slide, 12);
  }

  return p;
}

async function saveBlobToFile(blob, filePath) {
  if (typeof blob.save === "function") {
    await blob.save(filePath);
    return;
  }
  const ab = await blob.arrayBuffer();
  await fs.writeFile(filePath, Buffer.from(ab));
}

async function main() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const p = createDeck();
  for (let i = 0; i < p.slides.items.length; i += 1) {
    const preview = await p.export({ slide: p.slides.items[i], format: "png", scale: 1 });
    await saveBlobToFile(preview, path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`));
  }
  const pptx = await PresentationFile.exportPptx(p);
  const outPath = path.join(OUT_DIR, "adaptive_coverage_experiment_report.pptx");
  await pptx.save(outPath);
  console.log(outPath);
}

await main();

// 所有中英文内容都集中在这里。修改后刷新浏览器即可看到效果。
window.PORTFOLIO_CONTENT = {
  shared: {
    name: "Fangcai Zhao",
    initials: "FZ",
    cvHref: "./assets/Fangcai-Zhao-CV.pdf",
    links: [{ label: "GitHub", href: "https://github.com/ABan12" }],
  },

  en: {
    name: "Fangcai Zhao",
    pageTitle: "Fangcai Zhao — Personal Homepage",
    description: "Fangcai Zhao — AI systems, GPU computing, and efficient inference.",
    nav: ["About", "News", "Education", "Publications", "Projects", "Awards", "Contact"],
    role: "AI Systems · GPU Computing · Inference Optimization",
    affiliation: "Master’s Student",
    location: "Suzhou / Beijing, China",
    intro:
      "Hello 👋, I’m Fangcai Zhao. I am currently pursuing a professional master’s degree in Electronic Information (Artificial Intelligence and Smart Governance) at the School of Smart Governance, Renmin University of China, under the supervision of Professor Jing Wang from the School of Information. My research focuses on AI systems, high-performance computing, and efficient inference, with an ongoing interest in hardware–software co-optimization for real-world workloads.",
    availability: "Open to research conversations and collaboration.",
    cvLabel: "Download CV",
    sections: {
      newsEyebrow: "Recent updates",
      newsTitle: "News",
      educationEyebrow: "Academic background",
      educationTitle: "Education",
      publicationsEyebrow: "Research output",
      publicationsTitle: "Publications",
      projectsEyebrow: "Selected work",
      projectsTitle: "Projects",
      openSourceTitle: "Open-source PRs",
      notesTitle: "Notes",
      awardsEyebrow: "Recognition",
      awardsTitle: "Awards",
      contact: "Contact",
    },
    news: [
      {
        title: "Placed first for kv_rmsnorm_rope_cache on the Ascend track at the FlagOS 48-Hour Operator Bounty Challenge (Shanghai, Stop 5).",
        meta: "2026-09-10",
      },
      {
        title: "Received a National Second Prize in the Intelligent Computing Innovation Design Competition (Xiandao Cup), part of the National College Student Computer System Capability Competition.",
        meta: "2026-08-15",
      },
    ],
    openSource: [{
      title: "FlagGems-sglang #42 · chunk_cumsum Triton Kernel",
      description: "Implement Mamba’s chunk-wise cumulative sum in Triton, with generic and backend-specific implementations.",
      meta: "Merged · Opened 2026-09-04",
      href: "https://github.com/flagos-ai/FlagGems-sglang/pull/42",
    }],
    notes: [{
      title: "mini-SGLang",
      children: [{
        title: "Week 1 · Qwen3 Inference, Model Execution & KV Cache",
        meta: "2026-09-20",
        href: "./notes/mini-sglang/week01-qwen3-inference-organized.md",
      }],
    }, {
      title: "From FlashAttention to FA3 and FlashDecoding",
      meta: "2026-09-16",
      href: "./notes/flashattention-fa1-fa3-flashdecoding.md",
    }],
    notesEmpty: "Notes will be added here.",
    openSourceEmpty: "Open-source pull requests will be added here.",
    education: [
      {
        title:
          "Renmin University of China · School of Smart Governance · Professional Master’s in Electronic Information (Artificial Intelligence and Smart Governance)",
        meta: "Present",
        description: "Supervisor: Professor Jing Wang",
      },
    ],
    publications: [],
    projects: [],
    awards: [
      {
        title: "🏅 1st Place for kv_rmsnorm_rope_cache · Ascend Track",
        meta: "2026-09-10",
        description: "FlagOS 48-Hour Operator Bounty Challenge · Shanghai, Stop 5 — fused KV RMSNorm, RoPE, and cache operator.",
      },
      {
        title: "National Second Prize · Intelligent Computing Innovation Design Competition (Xiandao Cup)",
        meta: "2026-08-15",
        description: "National College Student Computer System Capability Competition. Project: Optimizing Qwen LLM inference serving on domestically developed accelerator cards.",
      },
    ],
    newsEmpty: "Recent publications, competitions, internships, and talks will appear here.",
    publicationsEmpty: "The publication list is being organized and will appear here.",
    projectsEmpty: "Research and engineering projects will be added here.",
    awardsEmpty: "Awards and recognitions will be added here.",
    contactText:
      "If you are interested in my papers or projects, please feel free to get in touch. I would also be very grateful for any research internship opportunities.",
    footer: "Designed for clarity. Built locally.",
  },

  zh: {
    name: "赵方材",
    pageTitle: "赵方材 — 个人主页",
    description: "赵方材的个人主页，关注人工智能系统、GPU 计算与高效推理。",
    nav: ["介绍", "动态", "教育", "论文", "项目", "荣誉", "联系"],
    role: "人工智能系统 · GPU 计算 · 推理优化",
    affiliation: "硕士",
    location: "中国 · 苏州 / 北京",
    intro:
      "你好，我是赵方材。我目前在中国人民大学智慧治理学院攻读电子信息（人工智能与智慧治理）专业硕士学位，师从王晶教授（中国人民大学信息学院）。我目前的研究方向主要是人工智能系统、高性能计算与高效推理，并持续探索软硬件协同优化在实际工作负载中的应用。",
    availability: "欢迎交流研究问题与合作想法。",
    cvLabel: "下载简历",
    sections: {
      newsEyebrow: "近期进展",
      newsTitle: "动态",
      educationEyebrow: "学习经历",
      educationTitle: "教育经历",
      publicationsEyebrow: "研究成果",
      publicationsTitle: "论文",
      projectsEyebrow: "代表工作",
      projectsTitle: "项目",
      openSourceTitle: "开源 PR",
      notesTitle: "笔记",
      awardsEyebrow: "荣誉与奖励",
      awardsTitle: "荣誉",
      contact: "联系",
    },
    news: [
      {
        title: "在 FlagOS 48小时算子赏金挑战赛（上海·第5站）中，获昇腾专项榜 kv_rmsnorm_rope_cache 算子第一。",
        meta: "2026-09-10",
        description: "KV RMSNorm RoPE 缓存融合。",
      },
      {
        title: "获全国大学生计算机系统能力大赛智能计算创新设计赛（先导杯）国家二等奖。",
        meta: "2026-08-15",
      },
    ],
    openSource: [{
      title: "FlagGems-sglang #42 · chunk_cumsum Triton 算子",
      description: "为 Mamba 的分块累积求和实现 Triton 算子，提供通用实现与不同硬件后端的专用实现。",
      meta: "已合并 · 提交于 2026-09-04",
      href: "https://github.com/flagos-ai/FlagGems-sglang/pull/42",
    }],
    notes: [{
      title: "mini-SGLang",
      children: [{
        title: "Week 1 · 跑通 Qwen3 推理，理解模型执行与 KV Cache",
        meta: "2026-09-20",
        href: "./notes/mini-sglang/week01-qwen3-inference-organized.md",
      }],
    }, {
      title: "从 FlashAttention 到 FA3，再到 FlashDecoding",
      meta: "2026-09-16",
      href: "./notes/flashattention-fa1-fa3-flashdecoding.md",
    }],
    notesEmpty: "笔记将在这里补充。",
    openSourceEmpty: "开源 PR 记录将在这里补充。",
    education: [
      {
        title: "中国人民大学 · 智慧治理学院 · 电子信息（人工智能与智慧治理）专业硕士",
        meta: "在读",
        description: "导师：王晶教授",
      },
    ],
    publications: [],
    projects: [],
    awards: [
      {
        title: "FlagOS 48小时算子赏金挑战赛 · 上海第5站",
        meta: "2026-09-10",
        description: "昇腾专项榜 · kv_rmsnorm_rope_cache（KV RMSNorm RoPE 缓存融合）第一。",
      },
      {
        title: "全国大学生计算机系统能力大赛 · 智能计算创新设计赛（先导杯）· 国家二等奖",
        meta: "2026-08-15",
        description: "基于国产加速卡的千问大模型推理服务优化",
      },
    ],
    newsEmpty: "论文、比赛、实习和报告等近期进展将在这里更新。",
    publicationsEmpty: "论文列表正在整理，之后会在这里更新。",
    projectsEmpty: "研究与工程项目将在这里更新。",
    awardsEmpty: "奖项与荣誉将在这里补充。",
    contactText:
      "如果对我的论文或者项目有兴趣，欢迎和我联系沟通。如果有研究型实习的机会，那将十分感激！",
    footer: "为清晰表达而设计，在本地构建。",
  },
};

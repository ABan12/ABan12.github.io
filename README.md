# Fangcai Zhao — Personal Homepage

Terminal 风格个人主页，支持中英文、明暗主题、笔记阅读和 KaTeX 公式。

## 本地预览

使用 Node.js 22 或更新版本运行 `npm run dev`，然后打开 http://127.0.0.1:5174/ 。
浏览器所需的 Markdown、公式渲染库和字体已保存在 assets/vendor，无需联网加载 CDN。

## 修改内容

- `content.js`：中英文个人信息、动态、教育、项目、荣誉、笔记和联系方式。
- `notes/`：Markdown 笔记正文。分组标题不设置链接，子笔记使用 children 数组。
- `styles.css`：主页样式；`note.css`：笔记阅读页样式。
- `assets/avatar.png`：头像；`assets/Fangcai-Zhao-CV.pdf`：简历文件。

公式支持正文中的 `$$...$$` 和 `$...$`；代码块内保留公式源码。
第八节的张量形状表及公式修复位于网站笔记副本中，重新复制外部文件时注意保留这些修改。

## 发布

仓库：https://github.com/ABan12/ABan12.github.io

线上地址：https://aban12.github.io/

向 main 分支提交后，GitHub Actions 自动运行 `npm run build`，并发布 dist/ 到 GitHub Pages。
Pages 的发布来源使用 GitHub Actions，不再使用旧 al-folio 构建。
旧版仍可从 Git 提交历史恢复。不要上传 node_modules、dist 或本地备份。

简历 PDF 目前仅有姓名占位，需替换为正式简历。

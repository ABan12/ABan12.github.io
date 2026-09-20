# 如何添加一篇笔记 / How to add a note

这是一篇使用说明示例，可以替换或删除，不是正式研究笔记。
This is an example guide, not a research note. You can replace or remove it.

## 1. 新建 Markdown 文件

在项目的 notes 文件夹内新建文件，例如 inference-notes.md。
正文可以包含标题、列表、代码块和链接。

Create a Markdown file in the notes folder, for example inference-notes.md.

## 2. 添加主页入口

打开 content.js，在 zh.notes 和 en.notes 中分别添加对应语言的条目：
Add an entry to both zh.notes and en.notes in content.js:

```js
{
  title: "推理优化学习笔记",
  description: "记录推理优化中的关键概念与实验思路。",
  href: "./notes/inference-notes.md",
}
```

两种语言的入口可以链接同一篇笔记，也可以分别链接中文和英文文件。
Both languages may link to the same file or to separate translated files.

## 3. 预览与构建

保存后刷新主页。点击笔记标题区域，会打开排版后的阅读页。
阅读页支持白天 / 夜晚切换，也可以点击「Markdown 原文」查看源文件。

Refresh the homepage after saving. Clicking a note opens a formatted reading page, with a theme toggle and a link to the Markdown source.

运行 npm run build 时，notes 文件夹会自动复制到 dist/notes，便于后续一起部署。
The build automatically copies the notes folder into dist/notes.

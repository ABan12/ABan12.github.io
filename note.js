const article = document.getElementById("note-content");
const themeButton = document.getElementById("reader-theme");
function updateTheme() {
  themeButton.textContent = document.documentElement.dataset.theme === "dark" ? "☀ 白天" : "☾ 夜晚";
}
themeButton.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("homepage-terminal-theme", next); } catch {}
  updateTheme();
});
updateTheme();

async function loadNote() {
  try {
    const file = new URLSearchParams(location.search).get("file");
    if (!file) throw new Error("未指定笔记文件。请返回主页选择一篇笔记。");
    const url = new URL(file, location.href);
    const notesRoot = new URL("./notes/", location.href);
    if (url.origin !== notesRoot.origin || !url.pathname.startsWith(notesRoot.pathname) || !url.pathname.endsWith(".md")) {
      throw new Error("笔记路径无效。请打开 notes 目录中的 Markdown 文件。");
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error("没有找到这篇笔记。请检查文件是否存在。");
    const markdown = await response.text();
    renderNoteMarkdown(markdown, article);
    for (const element of article.querySelectorAll("a[href], img[src]")) {
      const attribute = element.tagName === "A" ? "href" : "src";
      const value = element.getAttribute(attribute);
      if (value.startsWith("#")) continue;
      const resolved = new URL(value, url);
      if (element.tagName === "A" && resolved.origin === notesRoot.origin && resolved.pathname.startsWith(notesRoot.pathname) && resolved.pathname.endsWith(".md")) {
        const viewer = new URL("./note.html", location.href);
        viewer.searchParams.set("file", resolved.href);
        element.href = viewer.href;
      } else {
        element.setAttribute(attribute, resolved.href);
      }
    }
    const heading = article.querySelector("h1");
    document.title = `${heading ? heading.textContent : "笔记"} · Fangcai Zhao`;
    const raw = document.getElementById("raw-link");
    raw.href = url.href;
    raw.hidden = false;
  } catch (error) {
    article.replaceChildren();
    const title = document.createElement("h1");
    title.textContent = "暂时无法打开笔记";
    const message = document.createElement("p");
    message.textContent = error.message;
    article.append(title, message);
  }
}
loadNote();

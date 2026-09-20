const content = window.PORTFOLIO_CONTENT;
const shared = content.shared;

function text(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function createLink(link, className = "text-link") {
  const anchor = document.createElement("a");
  anchor.className = className;
  anchor.href = link.href;
  anchor.textContent = link.label;
  if (link.href.startsWith("http")) {
    anchor.target = "_blank";
    anchor.rel = "noreferrer";
  }
  return anchor;
}

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add("visible");
    });
  },
  { threshold: 0.12 },
);

function observeNewElements() {
  document.querySelectorAll(".reveal:not(.visible)").forEach((element) => observer.observe(element));
}

function renderList(targetId, items, emptyText) {
  const target = document.getElementById(targetId);
  target.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty-copy";
    empty.textContent = emptyText;
    target.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    if (targetId === "notes-list" && Array.isArray(item.children)) {
      const group = document.createElement("section");
      group.className = "note-group";
      const heading = document.createElement("h3");
      heading.textContent = item.title;
      const list = document.createElement("ul");
      list.className = "note-children";
      item.children.forEach((note) => {
        const entry = document.createElement("li");
        const link = document.createElement("a");
        link.href = `./note.html?file=${encodeURIComponent(note.href)}`;
        link.textContent = note.title;
        const date = document.createElement("time");
        date.textContent = note.meta || "";
        if (note.meta) date.dateTime = note.meta;
        entry.append(link, date);
        list.appendChild(entry);
      });
      group.append(heading, list);
      target.appendChild(group);
      return;
    }
    const row = document.createElement(item.href ? "a" : "article");
    row.className = "content-row";
    if (item.href) row.href = targetId === "notes-list"
      ? `./note.html?file=${encodeURIComponent(item.href)}`
      : item.href;

    const body = document.createElement("div");
    body.className = "content-row-main";
    const title = document.createElement("strong");
    title.textContent = targetId === "news-list" ? `🎉 ${item.title}` : item.title;
    body.appendChild(title);
    if (item.description) {
      const description = document.createElement("p");
      description.textContent = item.description;
      body.appendChild(description);
    }
    const meta = document.createElement("span");
    meta.textContent = item.meta || "";
    row.append(body, meta);
    target.appendChild(row);
  });
}

function renderLinks(language) {
  const heroLinks = document.getElementById("hero-links");
  heroLinks.replaceChildren();
  shared.links.forEach((link, index) => {
    heroLinks.appendChild(createLink(link, index === 0 ? "button-link primary" : "button-link"));
  });
  const cvLink = createLink(
    { label: content[language].cvLabel, href: shared.cvHref },
    "button-link",
  );
  cvLink.download = "Fangcai-Zhao-CV.pdf";
  heroLinks.appendChild(cvLink);

  const contactLinks = document.getElementById("contact-links");
  contactLinks.replaceChildren();
  shared.links.forEach((link) => contactLinks.appendChild(createLink(link, "button-link light")));
}

function renderPage(language) {
  const page = content[language];
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.title = page.pageTitle;
  document.querySelector('meta[name="description"]').setAttribute("content", page.description);

  text("nav-about", page.nav[0]);
  text("nav-news", page.nav[1]);
  text("nav-education", page.nav[2]);
  text("nav-projects", page.nav[4]);
  text("nav-open-source", page.sections.openSourceTitle);
  text("nav-notes", page.sections.notesTitle);
  text("nav-awards", page.nav[5]);
  text("nav-contact", page.nav[6]);
  text("wordmark-name", page.name);
  text("identity-name", page.name);
  text("about-title", language === "zh" ? "自我介绍" : "About Me");
  text("institution", language === "zh" ? "中国人民大学" : "Renmin University of China");
  text("school", language === "zh" ? "智慧治理学院" : "School of Smart Governance");
  text("footer-name", page.name);
  text("role", page.role);
  text("intro", page.intro);
  text("affiliation", page.affiliation);
  text("location", page.location);
  text("availability", page.availability);
  text("news-eyebrow", page.sections.newsEyebrow);
  text("news-title", page.sections.newsTitle);
  text("education-eyebrow", page.sections.educationEyebrow);
  text("education-title", page.sections.educationTitle);
  text("projects-eyebrow", page.sections.projectsEyebrow);
  text("projects-title", page.sections.projectsTitle);
  text("open-source-title", page.sections.openSourceTitle);
  text("notes-title", page.sections.notesTitle);
  text("awards-eyebrow", page.sections.awardsEyebrow);
  text("awards-title", page.sections.awardsTitle);
  text("contact-kicker", page.sections.contact);
  text("contact-text", page.contactText);
  text("footer-note", page.footer);

  renderLinks(language);
  renderList("news-list", page.news, page.newsEmpty);
  renderList("education-list", page.education, "");
  renderList("projects-list", page.projects, page.projectsEmpty);
  renderList("open-source-list", page.openSource, page.openSourceEmpty);
  renderList("notes-list", page.notes, page.notesEmpty);
  renderList("awards-list", page.awards, page.awardsEmpty);
  observeNewElements();
}

text("wordmark-initials", shared.initials);
text("year", new Date().getFullYear());

const themeToggle = document.getElementById("theme-toggle");
const savedTheme = localStorage.getItem("homepage-terminal-theme");
document.documentElement.dataset.theme = savedTheme === "light" ? "light" : "dark";

function updateThemeButton() {
  const dark = document.documentElement.dataset.theme === "dark";
  themeToggle.textContent = dark ? "☀ 白天" : "☾ 夜晚";
  themeToggle.setAttribute("aria-label", dark ? "切换到白天模式" : "切换到夜晚模式");
  themeToggle.title = dark ? "切换到白天模式" : "切换到夜晚模式";
}
updateThemeButton();

themeToggle.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("homepage-terminal-theme", next);
  updateThemeButton();
});

const languageToggle = document.getElementById("language-toggle");
let currentLanguage = localStorage.getItem("homepage-language") === "zh" ? "zh" : "en";

function updateLanguageButton() {
  languageToggle.textContent = currentLanguage === "en" ? "中" : "EN";
  languageToggle.setAttribute(
    "aria-label",
    currentLanguage === "en" ? "切换到中文" : "Switch to English",
  );
}

languageToggle.addEventListener("click", () => {
  currentLanguage = currentLanguage === "en" ? "zh" : "en";
  localStorage.setItem("homepage-language", currentLanguage);
  updateLanguageButton();
  renderPage(currentLanguage);
});

updateLanguageButton();
renderPage(currentLanguage);

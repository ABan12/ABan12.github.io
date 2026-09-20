// Tokenize math before Markdown can interpret underscores or +/- as markup.
// Fenced code and inline code remain source examples, not rendered equations.
function parseNoteMarkdown(markdown) {
  // Separate display equations from paragraphs before setext-heading/list rules
  // can consume their lines. Skip fenced and inline code examples.
  const normalized = markdown.replace(
    /(^ {0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?^ {0,3}\2[ \t]*(?=\n|$))|(`+)[\s\S]*?\3|(\$\$[\s\S]*?\$\$)/gm,
    (match) => match.startsWith("$$") ? `\n\n${match}\n\n` : match,
  );
  const formulas = [];
  const placeholder = (token) => {
    const id = formulas.push({ tex: token.text, display: token.display }) - 1;
    const tag = token.display ? "div" : "span";
    return `<${tag} data-note-math="${id}"></${tag}>`;
  };
  const parser = new marked.Marked({ extensions: [
    {
      name: "displayMath", level: "block",
      start(src) { const index = src.indexOf("\n$$"); return index < 0 ? undefined : index + 1; },
      tokenizer(src) {
        const match = /^\$\$([\s\S]+?)\$\$(?:[ \t]*\n)?/.exec(src);
        if (match) return { type: "displayMath", raw: match[0], text: match[1].trim(), display: true };
      },
      renderer: placeholder,
    },
    {
      name: "inlineMath", level: "inline",
      start(src) { return src.indexOf("$"); },
      tokenizer(src) {
        const display = /^\$\$([\s\S]+?)\$\$/.exec(src);
        if (display) return { type: "inlineMath", raw: display[0], text: display[1].trim(), display: true };
        const match = /^\$(?!\s|\$)((?:\\.|[^$\n\\])+?)\$(?!\d)/.exec(src);
        if (match) return { type: "inlineMath", raw: match[0], text: match[1].trim(), display: false };
      },
      renderer: placeholder,
    },
  ] });
  return { html: parser.parse(normalized), formulas };
}

function renderNoteMarkdown(markdown, target) {
  const { html, formulas } = parseNoteMarkdown(markdown);
  target.innerHTML = DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
  for (const element of target.querySelectorAll("[data-note-math]")) {
    const formula = formulas[Number(element.dataset.noteMath)];
    if (!formula) continue;
    katex.render(formula.tex, element, {
      displayMode: formula.display,
      throwOnError: false,
      trust: false,
      maxExpand: 1000,
      maxSize: 20,
    });
  }
}

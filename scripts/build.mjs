import { copyFile, cp, mkdir, readFile, rm } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = process.cwd();
const output = join(root, "dist");
const files = ["index.html", "styles.css", "content.js", "script.js", "note.html", "note.css", "note.js", "note-math.js"];
const assets = ["assets/avatar.png", "assets/Fangcai-Zhao-CV.pdf", "assets/minimal-light/style.css", "assets/minimal-light/LICENSE"];

await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });

for (const file of files) {
  await copyFile(join(root, file), join(output, file));
}

for (const asset of assets) {
  const target = join(output, asset);
  await mkdir(dirname(target), { recursive: true });
  await copyFile(join(root, asset), target);
}

const html = await readFile(join(output, "index.html"), "utf8");
await cp(join(root, "notes"), join(output, "notes"), { recursive: true });
await cp(join(root, "assets/vendor"), join(output, "assets/vendor"), { recursive: true });
for (const asset of ["styles.css", "content.js", "script.js"]) {
  if (!html.includes(asset)) throw new Error(`index.html 未引用 ${asset}`);
}

console.log(`Built ${files.length + assets.length} files into dist/`);

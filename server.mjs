import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const useDist = process.argv.includes("--dist");
const root = useDist ? join(process.cwd(), "dist") : process.cwd();
const port = Number(process.env.PORT || (useDist ? 4174 : 5174));

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".pdf": "application/pdf",
  ".md": "text/plain; charset=utf-8",
};

createServer(async (request, response) => {
  try {
    const url = new URL(request.url || "/", "http://127.0.0.1");
    const requestPath = url.pathname === "/" ? "index.html" : url.pathname.slice(1);
    const safePath = normalize(requestPath).replace(/^(\.\.[/\\])+/, "");
    const filePath = join(root, safePath);
    const fileStat = await stat(filePath);
    const finalPath = fileStat.isDirectory() ? join(filePath, "index.html") : filePath;
    const body = await readFile(finalPath);
    response.writeHead(200, {
      "Content-Type": contentTypes[extname(finalPath)] || "application/octet-stream",
      "Cache-Control": useDist ? "public, max-age=300" : "no-store",
    });
    response.end(body);
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    response.end("页面不存在");
  }
}).listen(port, "127.0.0.1", () => {
  console.log(`Local: http://127.0.0.1:${port}/`);
});

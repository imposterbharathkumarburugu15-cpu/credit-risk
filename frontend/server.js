const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = Number(process.env.PORT || 3000);
const API_HOST = process.env.API_HOST || "127.0.0.1";
const API_PORT = Number(process.env.API_PORT || 8000);
const ROOT = __dirname;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
};

const server = http.createServer((req, res) => {
  const urlPath = (req.url || "/").split("?")[0];

  // Proxy /api/* to FastAPI backend
  if (urlPath.startsWith("/api/") || urlPath === "/health") {
    const proxyReq = http.request(
      {
        hostname: API_HOST,
        port: API_PORT,
        path: req.url,
        method: req.method,
        headers: req.headers,
      },
      (proxyRes) => {
        res.writeHead(proxyRes.statusCode || 502, proxyRes.headers);
        proxyRes.pipe(res, { end: true });
      }
    );
    proxyReq.on("error", (err) => {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ detail: `Backend unavailable: ${err.message}` }));
    });
    req.pipe(proxyReq, { end: true });
    return;
  }

  // Serve static frontend files
  const safePath = urlPath === "/" ? "/index.html" : urlPath;
  const fullPath = path.normalize(path.join(ROOT, safePath));
  if (fullPath !== ROOT && !fullPath.startsWith(ROOT + path.sep)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(fullPath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not found");
      return;
    }
    const ext = path.extname(fullPath).toLowerCase();
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Abhaya frontend running at http://127.0.0.1:${PORT} (proxying /api -> http://${API_HOST}:${API_PORT})`);
});

# 部署指南：把校园事务智能助手变成可分享的在线链接

本系统 = **React 静态前端** + **FastAPI 后端 API**。要发给招聘人员一个"点开即用"的链接，
最佳做法是在线部署：前端放 Vercel（免费、自动 HTTPS），后端放 Render（免费）或 Railway。

> 好消息：**默认无需任何 API Key**。后端 `VLM_PROVIDER=mock` + 本地哈希向量 + Stub OCR
> 即可把"通知识别→抽取→待办→追踪"全链路跑通，演示完全够用。想接真实视觉大模型再填 Key。

---

## 0. 准备：推到 GitHub
（招聘人员/面试官也能直接看代码，强烈建议）

```bash
cd campus-assistant
git init
git add .
git commit -m "feat: 校园事务智能助手 MVP"
git branch -M main
git remote add origin https://github.com/<你的用户名>/campus-assistant.git
git push -u origin main
```
> 根目录 `.gitignore` 已忽略 `.env`、虚拟环境、数据库文件，不会泄露密钥。

---

## 1. 部署后端（Render，免费）
1. 打开 https://render.com ，用 GitHub 登录。
2. **New → Web Service**，选择本仓库。
3. 配置：
   - **Root Directory**：`backend`
   - **Runtime**：Docker（自动用 `backend/Dockerfile`）；或选 Python + 填 `Build: pip install -r requirements.txt`、`Start: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**：Free
   - **Health Check Path**：`/health`
4. **Environment（环境变量）**：
   - `CORS_ORIGINS` = `https://<你后面的 Vercel 域名>`（例如 `https://campus-assistant.vercel.app`；逗号或 JSON 数组都行）
   - 想持久化数据：在 Render 加一个 **PostgreSQL** 插件，并把 `DATABASE_URL` 填进去
   - （可选）`VLM_PROVIDER=dashscope` + `DASHSCOPE_API_KEY=sk-xxx` 接真实模型
5. 点击 **Deploy**。完成后记下后端公网地址，形如 `https://campus-assistant-backend.onrender.com`。

> 用 Railway 也行：连 GitHub 仓库，Root=`backend`，它会读 `backend/Procfile` 自动启动。

---

## 2. 部署前端（Vercel，免费）
1. 打开 https://vercel.com ，用 GitHub 登录。
2. **Add New → Project**，选择本仓库。
3. 配置：
   - **Root Directory**：`frontend`
   - Framework 会自动识别为 Vite；`vercel.json` 已写好构建/输出。
4. **Environment Variables（必须）**：
   - `VITE_API_BASE` = `https://<你的 Render 后端地址>/api`
     - 例：`https://campus-assistant-backend.onrender.com/api`
     - ⚠️ 一定要带 `/api` 后缀（前端接口路径是 `/documents`、`/notices`…，`/api` 由后端统一加）
5. **Deploy**。完成后得到前端地址，形如 `https://campus-assistant.vercel.app`。

---

## 3. 回填跨域（关键一步）
前端部署后域名才确定，回到 **Render** 控制台：
- 把后端环境变量 `CORS_ORIGINS` 改为你真正拿到的 Vercel 域名（如 `https://campus-assistant.vercel.app`）
- 保存后 Render 会自动重新部署。

---

## 4. 验收 & 分享
打开前端地址，试一下：粘贴一段课程通知文本 → 看抽取结果 → 待办看板出现任务。
把前端链接发给招聘人员即可，他们无需安装任何东西。

---

## 常见问题
- **后端免费版会休眠**：Render Free 一段时间无访问会暂停，首次打开可能慢几秒，属正常。
- **数据不持久**：Free 版磁盘/SQLite 在重启后可能清空。演示无所谓；要长期保留就接 Render PostgreSQL（`DATABASE_URL`）。
- **想本地跑**：见 `README.md` 与 `start.bat` / `start.sh`，无需部署。
- **自定义域名**：Vercel / Render 都支持绑定自己的域名（在各自控制台添加）。

## 其他平台速查
- **前端**还能放：Netlify、Cloudflare Pages、GitHub Pages（均只托管 `frontend/dist` 静态文件）。
- **后端**还能放：Fly.io、Railway、任意支持 Docker 的云；沿用 `backend/Dockerfile` 即可。

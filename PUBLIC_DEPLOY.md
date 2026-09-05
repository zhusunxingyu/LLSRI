# 低空物流无人机风险预测平台公网部署说明

## 目标

实现“只要打开一个公网网址即可访问平台”，访问者不需要启动本地模型接口，也不需要运行任何本地程序。

## 推荐部署方式

推荐使用支持 Python 后端的 Web Service 平台，例如 Render、Railway、Hugging Face Spaces Docker、阿里云轻量服务器或腾讯云轻量服务器。

不建议只部署为静态网页，因为当前平台需要调用 XGBoost 模型文件：

`outputs/models/xgboost_non_risk_operation_risk.joblib`

静态网页平台无法直接运行 Python 和 `.joblib` 模型。

## 已准备好的部署入口

公网服务入口：

`public_web_server.py`

该入口同时提供：

- `/`：网页首页
- `/predict`：XGBoost 风险预测接口
- `/health`：健康检查接口

## 已准备好的部署文件

- `Dockerfile`
- `requirements_public.txt`
- `render.yaml`
- `Procfile`
- `.dockerignore`

## Render 部署步骤

1. 将整个项目文件夹上传到 GitHub 仓库。
2. 打开 Render，选择 New Web Service。
3. 连接该 GitHub 仓库。
4. Environment 选择 Docker。
5. Health Check Path 填写 `/health`。
6. 点击 Deploy。
7. 部署完成后，Render 会生成一个公网网址。

## 注意事项

部署时必须保留以下文件和目录：

- `public_web_server.py`
- `model_api_server.py`
- `src/`
- `outputs/web/ai_risk_agent_route_planning_3d_webpage.html`
- `outputs/models/xgboost_non_risk_operation_risk.joblib`
- `requirements_public.txt`
- `Dockerfile`

如果缺少模型文件，网页可以打开，但 AI 风险预测接口会失败。

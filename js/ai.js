/* =====================================================================
 * TorchBlox — ai.js
 * DeepSeek API 集成：导入 PyTorch 代码 → AI 分析结构 → 转换为积木图。
 * - API Key 保存在 localStorage
 * - 调用 DeepSeek chat completions 接口（OpenAI 兼容格式）
 * - AI 返回 JSON（nodes + edges），解析后加载到画布
 * ===================================================================== */
(function () {
  'use strict';

  const LS_API_KEY = 'torchblox_deepseek_apikey';
  const API_URL = 'https://api.deepseek.com/chat/completions';
  const MODEL = 'deepseek-chat';

  // ---------- API Key 管理 ----------
  function getApiKey() {
    try { return localStorage.getItem(LS_API_KEY) || ''; } catch (e) { return ''; }
  }
  function setApiKey(key) {
    try { localStorage.setItem(LS_API_KEY, key || ''); } catch (e) {}
  }
  function hasApiKey() { return !!getApiKey(); }

  // ---------- 构建可用模块目录（给 AI 的提示） ----------
  function buildBlockCatalog() {
    const defs = window.BLOCK_DEFS || {};
    const skip = new Set(['input', 'output']);
    const lines = [];
    Object.keys(defs).forEach(type => {
      const d = defs[type];
      if (d._custom) return;       // 自定义模块不纳入（AI 不知其存在）
      if (skip.has(type)) return;  // input/output 单独说明
      const params = (d.paramSchema || []).map(f => f.key).join(', ');
      lines.push(`- ${type}: inputs=${d.inputs || 1}, outputs=${d.outputs || 1}, params=[${params}]`);
    });
    return lines.join('\n');
  }

  // ---------- 构建 system prompt ----------
  function buildSystemPrompt() {
    const catalog = buildBlockCatalog();
    return `你是一个 PyTorch 模型结构分析专家。用户会给你一段 PyTorch 代码（nn.Module 子类），你需要分析其网络结构，并将其转换为一个有向无环图（DAG），用 JSON 表示。

## 可用模块类型
${catalog}

## 特殊模块
- input: 网络输入节点，params: { "shape": "[batch, channels, ...]" }（根据 forward 的输入参数推断形状，默认 [1, 3, 224, 224]）
- output: 网络输出节点，无参数

## 输出 JSON 格式（严格遵守）
\`\`\`json
{
  "modelName": "类名",
  "nodes": [
    { "id": "n1", "type": "input", "params": { "shape": "[1, 3, 224, 224]" } },
    { "id": "n2", "type": "conv2d", "params": { "in_channels": 3, "out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3 } },
    { "id": "n3", "type": "output", "params": {} }
  ],
  "edges": [
    { "from": "n1", "to": "n2" },
    { "from": "n2", "to": "n3" }
  ]
}
\`\`\`

## 规则
1. 每个网络必须有且仅有一个 input 节点作为起点，至少一个 output 节点作为终点
2. 节点 id 用 n1, n2, n3... 递增命名
3. edges 中 from/to 是节点 id 字符串，表示数据流向（from 的输出 → to 的输入）
4. 对于多输入模块（如 add/concat 有 2 个输入），可以用多条 edge 指向同一节点，用 "port" 字段指定输入端口（0 或 1），不写则默认 0
5. 参数值必须是数字或字符串，不要用变量名
6. 残差连接、跳连用多条 edge 表示
7. 忽略 forward 中纯 Python 逻辑（如 if 判断），只保留 nn.Module 调用链
8. 如果代码中有自定义子模块（如 self.block = SomeBlock()），尝试递归展开为其内部结构；若无法展开则用最接近的内置模块替代
9. 只输出 JSON，不要输出任何其他文字解释

## 多输入模块示例（残差连接）
\`\`\`json
{
  "nodes": [
    { "id": "n1", "type": "input", "params": { "shape": "[1, 64, 56, 56]" } },
    { "id": "n2", "type": "conv2d", "params": { "in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1 } },
    { "id": "n3", "type": "add", "params": {} },
    { "id": "n4", "type": "output", "params": {} }
  ],
  "edges": [
    { "from": "n1", "to": "n2" },
    { "from": "n2", "to": "n3", "port": 1 },
    { "from": "n1", "to": "n3", "port": 0 },
    { "from": "n3", "to": "n4" }
  ]
}
\`\`\``;
  }

  // ---------- 调用 DeepSeek API ----------
  // onProgress(statusText) 用于更新 UI 加载状态
  async function analyzeCode(pyCode, onProgress) {
    const apiKey = getApiKey();
    if (!apiKey) {
      throw new Error('未设置 DeepSeek API Key，请先在设置中填写');
    }

    const systemPrompt = buildSystemPrompt();
    if (onProgress) onProgress('正在发送代码到 DeepSeek 分析…');

    const resp = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey
      },
      body: JSON.stringify({
        model: MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: '请分析以下 PyTorch 代码并转换为积木图 JSON：\n\n```python\n' + pyCode + '\n```' }
        ],
        temperature: 0.1,
        max_tokens: 4096,
        response_format: { type: 'json_object' }
      })
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => '');
      let msg = `API 请求失败 (HTTP ${resp.status})`;
      if (resp.status === 401) msg = 'API Key 无效或已过期，请检查设置';
      else if (resp.status === 429) msg = '请求过于频繁或额度不足，请稍后重试';
      else if (errText) {
        try { const j = JSON.parse(errText); if (j.error && j.error.message) msg += '：' + j.error.message; } catch (e) {}
      }
      throw new Error(msg);
    }

    if (onProgress) onProgress('正在解析 AI 返回的结构…');
    const data = await resp.json();
    const content = data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
    if (!content) {
      throw new Error('AI 返回内容为空');
    }

    return parseAIResponse(content);
  }

  // ---------- 解析 AI 返回的 JSON ----------
  function parseAIResponse(content) {
    // 尝试直接解析
    let jsonStr = content.trim();

    // 如果被 markdown 代码块包裹，提取内容
    const fenceMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (fenceMatch) {
      jsonStr = fenceMatch[1].trim();
    }

    let parsed;
    try {
      parsed = JSON.parse(jsonStr);
    } catch (e) {
      // 尝试提取第一个 { ... } 块
      const objMatch = content.match(/\{[\s\S]*\}/);
      if (objMatch) {
        try { parsed = JSON.parse(objMatch[0]); } catch (e2) {
          throw new Error('AI 返回的内容无法解析为 JSON：' + content.slice(0, 200));
        }
      } else {
        throw new Error('AI 返回的内容无法解析为 JSON：' + content.slice(0, 200));
      }
    }

    return normalizeGraph(parsed);
  }

  // ---------- 规范化为 TorchBlox 内部格式 ----------
  function normalizeGraph(parsed) {
    if (!parsed || !Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
      throw new Error('AI 返回的 JSON 缺少 nodes 或 edges 字段');
    }

    const defs = window.BLOCK_DEFS || {};
    const validTypes = new Set(Object.keys(defs));
    const warnings = [];

    // 规范化节点
    const nodes = [];
    const idMap = {}; // 原始 id → 规范化 id（确保唯一）
    parsed.nodes.forEach((n, i) => {
      if (!n.id) n.id = 'n' + (i + 1);
      const newId = 'n' + (i + 1);
      idMap[n.id] = newId;
      const type = n.type;
      if (!validTypes.has(type)) {
        warnings.push('未知模块类型「' + type + '」，已替换为 Identity');
      }
      const safeType = validTypes.has(type) ? type : 'identity';
      const def = defs[safeType];
      // 合并默认参数与 AI 给的参数
      const defaultP = def && def.defaultParams ? JSON.parse(JSON.stringify(def.defaultParams)) : {};
      const params = Object.assign({}, defaultP, n.params || {});
      nodes.push({
        id: newId,
        type: safeType,
        x: 0, y: 0,  // 由 autoLayout 排列
        params: params,
        label: def ? def.name : type
      });
    });

    // 规范化边
    const edges = [];
    const edgeIdSet = new Set();
    parsed.edges.forEach((e, i) => {
      const fromId = idMap[e.from] || e.from;
      const toId = idMap[e.to] || e.to;
      const fromNode = nodes.find(n => n.id === fromId);
      const toNode = nodes.find(n => n.id === toId);
      if (!fromNode || !toNode) {
        warnings.push('跳过无效连线：' + e.from + ' → ' + e.to);
        return;
      }
      const port = Number(e.port || 0);
      const edgeId = 'e' + (i + 1);
      if (edgeIdSet.has(edgeId)) return;
      edgeIdSet.add(edgeId);
      edges.push({
        id: edgeId,
        from: { nodeId: fromId, port: 0 },  // 输出端口默认 0
        to: { nodeId: toId, port: port }
      });
    });

    // 确保有 input 节点
    if (!nodes.find(n => n.type === 'input')) {
      nodes.unshift({
        id: 'n0', type: 'input', x: 0, y: 0,
        params: { shape: '[1, 3, 224, 224]' }, label: 'Input'
      });
      idMap['n0'] = 'n0';
    }
    // 确保有 output 节点
    if (!nodes.find(n => n.type === 'output')) {
      const lastId = nodes[nodes.length - 1].id;
      nodes.push({
        id: 'n_out', type: 'output', x: 0, y: 0, params: {}, label: 'Output'
      });
      edges.push({ id: 'e_out', from: { nodeId: lastId, port: 0 }, to: { nodeId: 'n_out', port: 0 } });
    }

    return {
      modelName: parsed.modelName || 'ImportedModel',
      nodes: nodes,
      edges: edges,
      warnings: warnings
    };
  }

  // ---------- 导出 ----------
  window.AI = {
    getApiKey,
    setApiKey,
    hasApiKey,
    analyzeCode,
    parseAIResponse,
    normalizeGraph,
    buildBlockCatalog,
    buildSystemPrompt
  };
})();

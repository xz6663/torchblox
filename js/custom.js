/* =====================================================================
 * TorchBlox — custom.js
 * 自定义模块管理 + 用户模板持久化。
 * - 自定义模块：把画布中的若干节点打包成一个可复用模块，注册到 BLOCK_DEFS。
 *   内部子图保存在 localStorage，shapeFn/codegenInit/codegenForward 动态生成。
 * - 用户模板：把当前画布保存为模板，下次可一键加载。
 * ===================================================================== */
(function () {
  'use strict';

  const LS_BLOCKS = 'torchblox_custom_blocks';
  const LS_TEMPLATES = 'torchblox_user_templates';

  // ---------- 工具 ----------
  function deepClone(o) { return JSON.parse(JSON.stringify(o)); }
  function uid(prefix) {
    return prefix + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }

  // ---------- 读取/写入 localStorage ----------
  function loadArr(key) {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch (e) { return []; }
  }
  function saveArr(key, arr) {
    try { localStorage.setItem(key, JSON.stringify(arr)); } catch (e) {}
  }

  // ---------- 自定义模块注册表 ----------
  // 每个自定义模块定义：
  // {
  //   type: 'custom_xxx',
  //   name, icon, category: 'custom',
  //   inputs, outputs,
  //   paramSchema: [...],          // 用户可配置参数
  //   defaultParams: {...},
  //   desc: { summary, role, uses },
  //   graph: { nodes, edges },     // 内部子图（节点 type 为标准模块）
  //   inputBindings: [nodeId,...], // 第 i 个外部输入端口 → 内部哪个节点的 input 端口 0
  //   outputBindings: [nodeId,...] // 第 i 个输出 ← 内部哪个节点的 output
  // }

  function getCustomBlocks() { return loadArr(LS_BLOCKS); }
  function saveCustomBlocks(arr) { saveArr(LS_BLOCKS, arr); }

  // 注册一个自定义模块到 BLOCK_DEFS（运行时）
  function registerBlockDef(cdef) {
    if (!window.BLOCK_DEFS) window.BLOCK_DEFS = {};
    if (!window.BLOCK_VAR_PREFIX) window.BLOCK_VAR_PREFIX = {};
    if (!window.BLOCK_EXPECTED_RANK) window.BLOCK_EXPECTED_RANK = {};
    if (!window.BLOCK_CATEGORIES) window.BLOCK_CATEGORIES = [];

    // 确保有 custom 分类
    if (!window.BLOCK_CATEGORIES.find(c => c.key === 'custom')) {
      window.BLOCK_CATEGORIES.push({ key: 'custom', label: '自定义模块' });
    }

    const type = cdef.type;
    // shapeFn：运行内部子图推断，取 outputBindings 节点的输出形状
    const shapeFn = function (inputShapes, params) {
      // 克隆内部子图，把 inputBindings 对应节点的 shape 设为外部输入
      const g = deepClone(cdef.graph);
      const nodeMap = {};
      g.nodes.forEach(n => { nodeMap[n.id] = n; });
      // 为内部 input 节点注入外部形状
      cdef.inputBindings.forEach((nodeId, i) => {
        const n = nodeMap[nodeId];
        if (n && n.type === 'input') {
          const shp = inputShapes[i] || [null];
          n.params = n.params || {};
          n.params.shape = '[' + shp.map(v => v == null ? '?' : v).join(', ') + ']';
        }
      });
      // 把自定义参数注入到子图中对应节点（通过 paramMapping）
      if (cdef.paramMapping) {
        cdef.paramMapping.forEach(m => {
          const n = nodeMap[m.nodeId];
          if (n) n.params[m.nodeParam] = params[m.blockParam];
        });
      }
      const res = window.ShapeInference.infer(g.nodes, g.edges);
      // 取第一个 outputBinding 的输出
      const outNodeId = cdef.outputBindings[0];
      const ns = res.nodeShapes[outNodeId];
      return (ns && ns.output) ? ns.output.slice() : [null];
    };

    // codegenInit：返回子模块类名，实际类定义在 codegen 阶段生成
    const codegenInit = function (p, varName, inputShapes) {
      // 子模块类名：CustomXxx，每个实例共用一个类定义
      const cls = cdef.className || ('Custom' + cdef.name.replace(/[^a-zA-Z0-9]/g, ''));
      // 记录需要生成的子类（由 codegen.js 读取）
      if (!window._pendingCustomClasses) window._pendingCustomClasses = {};
      if (!window._pendingCustomClasses[cls]) {
        window._pendingCustomClasses[cls] = cdef;
      }
      return cls + '()';
    };

    // codegenForward：调用 self.varName(x)
    const codegenForward = function (iv, p, varName) {
      if (iv.length === 1) return `self.${varName}(${iv[0]})`;
      return `self.${varName}(${iv.join(', ')})`;
    };

    window.BLOCK_DEFS[type] = {
      type, name: cdef.name, icon: cdef.icon, category: 'custom',
      inputs: cdef.inputs, outputs: cdef.outputs,
      defaultParams: deepClone(cdef.defaultParams || {}),
      paramSchema: deepClone(cdef.paramSchema || []),
      desc: cdef.desc || { summary: '用户自定义模块', role: '由画布子图打包而来', uses: '可复用的自定义组合模块' },
      shapeFn, codegenInit, codegenForward,
      _custom: true   // 标记为自定义模块
    };
    window.BLOCK_VAR_PREFIX[type] = cdef.varPrefix || 'cust';
    window.BLOCK_EXPECTED_RANK[type] = null;
  }

  // 从 localStorage 加载所有自定义模块并注册
  function loadAndRegisterAll() {
    // 始终确保有 custom 分类（即使 localStorage 为空，侧边栏也要显示空分类）
    if (!window.BLOCK_CATEGORIES) window.BLOCK_CATEGORIES = [];
    if (!window.BLOCK_CATEGORIES.find(c => c.key === 'custom')) {
      window.BLOCK_CATEGORIES.push({ key: 'custom', label: '自定义模块' });
    }
    const arr = getCustomBlocks();
    arr.forEach(cdef => registerBlockDef(cdef));
  }

  // 新建自定义模块
  // packConfig: { name, icon, nodes, edges, inputBindings, outputBindings, paramSchema, paramMapping, desc }
  function createCustomBlock(cfg) {
    const type = 'custom_' + cfg.name.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase() + '_' + uid('').slice(-4);
    const cdef = {
      type,
      name: cfg.name,
      icon: cfg.icon || '🧩',
      className: 'Custom' + cfg.name.replace(/[^a-zA-Z0-9]/g, ''),
      varPrefix: 'cust',
      inputs: cfg.inputs || 1,
      outputs: cfg.outputs || 1,
      paramSchema: cfg.paramSchema || [],
      defaultParams: (cfg.paramSchema || []).reduce((o, f) => { o[f.key] = f.default; return o; }, {}),
      desc: cfg.desc || { summary: '用户自定义模块：' + cfg.name, role: '由画布子图打包而来', uses: '可复用的自定义组合模块' },
      graph: { nodes: deepClone(cfg.nodes), edges: deepClone(cfg.edges) },
      inputBindings: cfg.inputBindings || [],
      outputBindings: cfg.outputBindings || [],
      paramMapping: cfg.paramMapping || []
    };
    const arr = getCustomBlocks();
    arr.push(cdef);
    saveCustomBlocks(arr);
    registerBlockDef(cdef);
    return cdef;
  }

  function deleteCustomBlock(type) {
    let arr = getCustomBlocks();
    arr = arr.filter(c => c.type !== type);
    saveCustomBlocks(arr);
    if (window.BLOCK_DEFS) delete window.BLOCK_DEFS[type];
    if (window.BLOCK_VAR_PREFIX) delete window.BLOCK_VAR_PREFIX[type];
    if (window.BLOCK_EXPECTED_RANK) delete window.BLOCK_EXPECTED_RANK[type];
  }

  // ---------- 用户模板 ----------
  function getUserTemplates() { return loadArr(LS_TEMPLATES); }
  function saveUserTemplates(arr) { saveArr(LS_TEMPLATES, arr); }

  function saveAsTemplate(name, nodes, edges, modelName) {
    const tpl = {
      key: 'user_' + uid(''),
      name: name,
      modelName: modelName || 'MyModel',
      nodes: deepClone(nodes),
      edges: deepClone(edges),
      _user: true
    };
    const arr = getUserTemplates();
    arr.push(tpl);
    saveUserTemplates(arr);
    return tpl;
  }

  function deleteUserTemplate(key) {
    let arr = getUserTemplates();
    arr = arr.filter(t => t.key !== key);
    saveUserTemplates(arr);
  }

  // ---------- 导出 ----------
  window.Custom = {
    getCustomBlocks,
    createCustomBlock,
    deleteCustomBlock,
    loadAndRegisterAll,
    registerBlockDef,
    getUserTemplates,
    saveAsTemplate,
    deleteUserTemplate
  };
})();

/* =====================================================================
 * TorchBlox — app.js
 * 全局状态 TB、事件总线、模块库渲染、属性面板、代码预览、工具栏联动。
 * ===================================================================== */
(function () {
  'use strict';

  // ---------- 工具函数 ----------
  function uid(prefix) {
    return prefix + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }
  function deepClone(o) { return JSON.parse(JSON.stringify(o)); }
  function debounce(fn, ms) {
    let t = null;
    return function () {
      clearTimeout(t);
      const args = arguments, self = this;
      t = setTimeout(() => fn.apply(self, args), ms);
    };
  }

  // ---------- 事件总线 ----------
  const listeners = {};
  function on(event, cb) {
    (listeners[event] = listeners[event] || []).push(cb);
  }
  function emit(event, data) {
    (listeners[event] || []).forEach(cb => { try { cb(data); } catch (e) { console.error(e); } });
  }

  // ---------- 全局状态 ----------
  const TB = {
    state: {
      nodes: [],
      edges: [],
      selectedNodeId: null,
      selectedEdgeId: null,
      selectedNodeIds: [],        // 多选节点集合
      selectedLibType: null,      // 侧边栏点击预览的模块类型
      history: [],
      historyIndex: -1,
      modelName: 'MyModel',
      includeTrainingLoop: false,
      pan: { x: 0, y: 0 },
      zoom: 1,
      nodeShapes: {},
      shapeErrors: []
    },
    on, emit,
    toast
  };

  // ---------- 历史快照 ----------
  function snapshot() {
    return {
      nodes: deepClone(TB.state.nodes),
      edges: deepClone(TB.state.edges),
      modelName: TB.state.modelName,
      includeTrainingLoop: TB.state.includeTrainingLoop
    };
  }
  function pushHistory() {
    // 丢弃 redo 分支
    TB.state.history = TB.state.history.slice(0, TB.state.historyIndex + 1);
    TB.state.history.push(snapshot());
    if (TB.state.history.length > 100) TB.state.history.shift();
    TB.state.historyIndex = TB.state.history.length - 1;
    updateUndoRedo();
  }
  function restore(snap) {
    TB.state.nodes = deepClone(snap.nodes);
    TB.state.edges = deepClone(snap.edges);
    TB.state.modelName = snap.modelName;
    TB.state.includeTrainingLoop = snap.includeTrainingLoop;
    TB.state.selectedNodeId = null;
    TB.state.selectedEdgeId = null;
    emit('graphChanged');
    emit('selectChanged');
  }
  function undo() {
    if (TB.state.historyIndex <= 0) return;
    TB.state.historyIndex -= 1;
    restore(TB.state.history[TB.state.historyIndex]);
    updateUndoRedo();
  }
  function redo() {
    if (TB.state.historyIndex >= TB.state.history.length - 1) return;
    TB.state.historyIndex += 1;
    restore(TB.state.history[TB.state.historyIndex]);
    updateUndoRedo();
  }

  // ---------- 状态操作 ----------
  function addNode(type, x, y, params) {
    const def = window.BLOCK_DEFS[type];
    if (!def) return null;
    const node = {
      id: uid('n'),
      type,
      x: Math.round(x),
      y: Math.round(y),
      params: params ? deepClone(params) : deepClone(def.defaultParams),
      label: def.name
    };
    TB.state.nodes.push(node);
    TB.state.selectedNodeId = node.id;
    TB.state.selectedEdgeId = null;
    pushHistory();
    emit('graphChanged');
    emit('selectChanged');
    return node;
  }
  function removeNode(id) {
    TB.state.nodes = TB.state.nodes.filter(n => n.id !== id);
    TB.state.edges = TB.state.edges.filter(e => e.from.nodeId !== id && e.to.nodeId !== id);
    if (TB.state.selectedNodeId === id) TB.state.selectedNodeId = null;
    pushHistory();
    emit('graphChanged');
    emit('selectChanged');
  }
  function moveNode(id, x, y) {
    const n = TB.state.nodes.find(n => n.id === id);
    if (n) { n.x = Math.round(x); n.y = Math.round(y); pushHistory(); emit('graphChanged'); }
  }
  function updateNodeParam(id, key, value, recordHistory) {
    const n = TB.state.nodes.find(n => n.id === id);
    if (!n) return;
    n.params[key] = value;
    if (recordHistory !== false) pushHistory();
    emit('graphChanged');
  }
  function addEdge(from, to) {
    // 校验：不能自连
    if (from.nodeId === to.nodeId) { toast('不能连接到自身', 'error'); return null; }
    // 一个输入端口只能连一条边（已在 editor 移除，这里再保险）
    const exist = TB.state.edges.find(e => e.to.nodeId === to.nodeId && e.to.port === to.port);
    if (exist) TB.state.edges = TB.state.edges.filter(e => e !== exist);
    const edge = { id: uid('e'), from, to };
    TB.state.edges.push(edge);
    pushHistory();
    emit('graphChanged');
    return edge;
  }
  function removeEdge(id, skipHistory) {
    TB.state.edges = TB.state.edges.filter(e => e.id !== id);
    if (TB.state.selectedEdgeId === id) TB.state.selectedEdgeId = null;
    if (skipHistory !== true) pushHistory();
    emit('graphChanged');
    emit('selectChanged');
  }
  function selectNode(id) {
    TB.state.selectedNodeId = id;
    if (id) { TB.state.selectedEdgeId = null; TB.state.selectedLibType = null; }
    emit('selectChanged');
  }
  function selectEdge(id) {
    TB.state.selectedEdgeId = id;
    if (id) { TB.state.selectedNodeId = null; TB.state.selectedLibType = null; }
    emit('selectChanged');
  }
  function selectLibType(type) {
    TB.state.selectedLibType = type;
    if (type) { TB.state.selectedNodeId = null; TB.state.selectedEdgeId = null; }
    emit('selectChanged');
  }
  function clearCanvas() {
    TB.state.nodes = [];
    TB.state.edges = [];
    TB.state.selectedNodeId = null;
    TB.state.selectedEdgeId = null;
    TB.state.nodeShapes = {};
    TB.state.shapeErrors = [];
    pushHistory();
    emit('graphChanged');
    emit('selectChanged');
  }
  function loadTemplate(tpl) {
    TB.state.nodes = deepClone(tpl.nodes);
    TB.state.edges = deepClone(tpl.edges);
    // 自动布局：按拓扑分层整齐排列，避免重叠
    autoLayout(TB.state.nodes, TB.state.edges);
    TB.state.selectedNodeId = null;
    TB.state.selectedEdgeId = null;
    TB.state.selectedLibType = null;
    if (tpl.modelName) TB.state.modelName = tpl.modelName;
    pushHistory();
    emit('graphChanged');
    emit('selectChanged');
    if (window.Editor && window.Editor.fitView) setTimeout(() => window.Editor.fitView(), 50);
  }

  // ---------- 自动布局（分层排列，保证不重叠） ----------
  function autoLayout(nodes, edges) {
    if (!nodes || nodes.length === 0) return;
    const NODE_W = 160, NODE_H = 60;
    const COL_GAP = 80;   // 列间距（节点间空隙）
    const ROW_GAP = 50;   // 行间距（节点间空隙）
    const COL_W = NODE_W + COL_GAP;
    const ROW_H = NODE_H + ROW_GAP;
    const START_X = 40, START_Y = 40;

    // 构建邻接表与入度
    const inDeg = {}, outAdj = {};
    nodes.forEach(n => { inDeg[n.id] = 0; outAdj[n.id] = []; });
    edges.forEach(e => {
      if (outAdj[e.from.nodeId] && inDeg[e.to.nodeId] != null) {
        outAdj[e.from.nodeId].push(e.to.nodeId);
        inDeg[e.to.nodeId]++;
      }
    });

    // 最长路径分层（Kahn 拓扑）
    const layer = {};
    const queue = [];
    nodes.forEach(n => {
      if (inDeg[n.id] === 0) { layer[n.id] = 0; queue.push(n.id); }
    });
    const inDegWork = Object.assign({}, inDeg);
    while (queue.length) {
      const id = queue.shift();
      const l = layer[id] || 0;
      (outAdj[id] || []).forEach(next => {
        layer[next] = Math.max(layer[next] != null ? layer[next] : 0, l + 1);
        inDegWork[next]--;
        if (inDegWork[next] === 0) queue.push(next);
      });
    }
    // 环中的节点兜底
    nodes.forEach(n => { if (layer[n.id] == null) layer[n.id] = 0; });

    // 按层分组
    const layers = {};
    let maxLayer = 0;
    nodes.forEach(n => {
      const l = layer[n.id];
      if (!layers[l]) layers[l] = [];
      layers[l].push(n);
      if (l > maxLayer) maxLayer = l;
    });

    // 居中排列：每层内的节点垂直居中于该层节点组的中心
    // 先计算每层节点数，确定最大层数对应的最大行数
    let maxRowCount = 0;
    Object.keys(layers).forEach(l => {
      if (layers[l].length > maxRowCount) maxRowCount = layers[l].length;
    });

    Object.keys(layers).sort((a, b) => Number(a) - Number(b)).forEach(l => {
      const layerNodes = layers[l];
      const rowCount = layerNodes.length;
      // 垂直居中：以 maxRowCount 为基准居中
      const totalH = rowCount * ROW_H;
      const centerOffset = (maxRowCount * ROW_H - totalH) / 2;
      layerNodes.forEach((n, i) => {
        n.x = START_X + Number(l) * COL_W;
        n.y = START_Y + centerOffset + i * ROW_H;
      });
    });
  }
  TB.autoLayout = autoLayout;

  TB.addNode = addNode;
  TB.removeNode = removeNode;
  TB.moveNode = moveNode;
  TB.updateNodeParam = updateNodeParam;
  TB.addEdge = addEdge;
  TB.removeEdge = removeEdge;
  TB.selectNode = selectNode;
  TB.selectEdge = selectEdge;
  TB.selectLibType = selectLibType;
  TB.pushHistory = pushHistory;
  TB.undo = undo;
  TB.redo = redo;
  TB.clearCanvas = clearCanvas;
  TB.loadTemplate = loadTemplate;

  // ---------- 多选清空 ----------
  function clearMultiSelection() {
    TB.state.selectedNodeIds = [];
  }
  TB.clearMultiSelection = clearMultiSelection;

  // ---------- 打包为自定义模块 ----------
  // 将选中的节点（selectedNodeIds 或全部）打包为一个自定义模块
  function packCustomBlock(config) {
    // config: { name, icon, paramSchema, paramMapping, desc }
    // 收集选中节点与内部边
    const selIds = new Set(TB.state.selectedNodeIds && TB.state.selectedNodeIds.length
      ? TB.state.selectedNodeIds
      : TB.state.nodes.map(n => n.id));
    const subNodes = TB.state.nodes.filter(n => selIds.has(n.id)).map(n => deepClone(n));
    const subEdges = TB.state.edges
      .filter(e => selIds.has(e.from.nodeId) && selIds.has(e.to.nodeId))
      .map(e => deepClone(e));

    if (subNodes.length === 0) { toast('没有可打包的节点', 'error'); return null; }

    // 识别输入/输出端口绑定：
    // 内部 input 节点 → 外部输入端口
    // 内部 output 节点的上游 → 外部输出
    const inputBindings = subNodes.filter(n => n.type === 'input').map(n => n.id);
    const outputNodes = subNodes.filter(n => n.type === 'output');
    const outputBindings = outputNodes.map(n => {
      // 找到连到该 output 的边的上游节点
      const ein = subEdges.find(e => e.to.nodeId === n.id);
      return ein ? ein.from.nodeId : null;
    }).filter(x => x != null);

    // 若没有 input 节点，自动添加一个并连接到子图的入口节点
    if (inputBindings.length === 0) {
      const newId = uid('n');
      subNodes.unshift({ id: newId, type: 'input', x: 0, y: 0, params: { shape: '[1, 3, 224, 224]' }, label: 'Input' });
      inputBindings.push(newId);
      // 找到子图中没有内部入边的节点（入口节点），把 input 连到它们的 0 号端口
      const hasIncoming = new Set();
      subEdges.forEach(e => hasIncoming.add(e.to.nodeId));
      subNodes.forEach(n => {
        if (n.id === newId) return;
        if (n.type === 'output') return;
        if (!hasIncoming.has(n.id)) {
          subEdges.push({ id: uid('e'), from: { nodeId: newId, port: 0 }, to: { nodeId: n.id, port: 0 } });
        }
      });
    }
    // 若没有 output 节点，取子图中没有内部出边的节点作为输出
    if (outputBindings.length === 0) {
      const hasOutgoing = new Set();
      subEdges.forEach(e => hasOutgoing.add(e.from.nodeId));
      const exits = subNodes.filter(n => n.type !== 'input' && n.type !== 'output' && !hasOutgoing.has(n.id));
      if (exits.length > 0) {
        outputBindings.push(exits[exits.length - 1].id);
      } else {
        // 兜底：取最后一个非 input 节点
        const last = [...subNodes].reverse().find(n => n.type !== 'input');
        if (last) outputBindings.push(last.id);
      }
    }

    const cdef = window.Custom.createCustomBlock({
      name: config.name,
      icon: config.icon || '🧩',
      desc: config.desc || { summary: '自定义模块：' + config.name, role: '由画布子图打包而来', uses: '可复用的组合模块' },
      nodes: subNodes,
      edges: subEdges,
      inputs: inputBindings.length,
      outputs: outputBindings.length,
      inputBindings,
      outputBindings,
      paramSchema: config.paramSchema || [],
      paramMapping: config.paramMapping || []
    });

    // 刷新侧边栏
    renderLibrary('');
    toast('已创建自定义模块：' + config.name, 'success');
    // 清空多选
    TB.state.selectedNodeIds = [];
    emit('selectChanged');
    if (window.Editor) window.Editor.render();
    return cdef;
  }
  TB.packCustomBlock = packCustomBlock;

  // ---------- 保存为模板 ----------
  function saveCurrentAsTemplate(name) {
    if (TB.state.nodes.length === 0) { toast('画布为空，无法保存', 'error'); return null; }
    const tpl = window.Custom.saveAsTemplate(name, TB.state.nodes, TB.state.edges, TB.state.modelName);
    toast('已保存模板：' + name, 'success');
    // 刷新模板下拉
    refreshTemplateMenu();
    return tpl;
  }
  TB.saveCurrentAsTemplate = saveCurrentAsTemplate;

  // 刷新模板下拉菜单（含用户模板）
  function refreshTemplateMenu() {
    const tplMenu = document.getElementById('tb-tpl-menu');
    if (!tplMenu) return;
    const userTpls = window.Custom.getUserTemplates();
    let html = TEMPLATES.map(t => `<div class="tb-menu-item" data-key="${t.key}">${t.name}</div>`).join('');
    if (userTpls.length > 0) {
      html += '<div class="tb-menu-sep"></div>';
      html += '<div class="tb-menu-label">用户模板</div>';
      userTpls.forEach(t => {
        html += `<div class="tb-menu-item" data-key="${t.key}" data-user="1">${escapeHtml(t.name)}<span class="tb-menu-del" data-del="${t.key}" title="删除">✕</span></div>`;
      });
    }
    html += '<div class="tb-menu-sep"></div>';
    html += '<div class="tb-menu-item" id="tb-save-tpl">💾 保存为模板…</div>';
    html += '<div class="tb-menu-item" id="tb-clear-canvas">清空画布</div>';
    tplMenu.innerHTML = html;

    // 绑定内置模板
    tplMenu.querySelectorAll('.tb-menu-item[data-key]:not([data-user])').forEach(it => {
      it.addEventListener('click', () => {
        const t = TEMPLATES.find(x => x.key === it.dataset.key);
        if (t) { TB.loadTemplate(t); toast('已加载模板：' + t.name, 'success'); }
        closeAllMenus();
      });
    });
    // 绑定用户模板
    tplMenu.querySelectorAll('.tb-menu-item[data-user]').forEach(it => {
      it.addEventListener('click', (e) => {
        if (e.target.classList.contains('tb-menu-del')) return;
        const userTpls = window.Custom.getUserTemplates();
        const t = userTpls.find(x => x.key === it.dataset.key);
        if (t) { TB.loadTemplate(t); toast('已加载用户模板：' + t.name, 'success'); }
        closeAllMenus();
      });
    });
    // 删除用户模板
    tplMenu.querySelectorAll('.tb-menu-del').forEach(it => {
      it.addEventListener('click', (e) => {
        e.stopPropagation();
        const key = it.dataset.del;
        if (confirm('确定删除该用户模板？')) {
          window.Custom.deleteUserTemplate(key);
          refreshTemplateMenu();
          toast('已删除模板', 'info');
        }
      });
    });
    // 保存为模板
    const saveBtn = tplMenu.querySelector('#tb-save-tpl');
    if (saveBtn) saveBtn.addEventListener('click', () => {
      closeAllMenus();
      showSaveTemplateDialog();
    });
    // 清空
    const clr = tplMenu.querySelector('#tb-clear-canvas');
    if (clr) clr.addEventListener('click', () => { TB.clearCanvas(); closeAllMenus(); toast('画布已清空', 'info'); });
  }

  // ---------- Toast ----------
  function toast(msg, type) {
    let host = document.getElementById('tb-toast-host');
    if (!host) return;
    const el = document.createElement('div');
    el.className = 'tb-toast ' + (type || 'info');
    el.textContent = msg;
    host.appendChild(el);
    setTimeout(() => el.classList.add('show'), 10);
    setTimeout(() => {
      el.classList.remove('show');
      setTimeout(() => { if (el.parentNode) el.parentNode.removeChild(el); }, 250);
    }, 2200);
  }
  TB.toast = toast;

  // ---------- 模块库渲染 ----------
  function renderLibrary(filter) {
    const host = document.getElementById('tb-library');
    if (!host) return;
    host.innerHTML = '';
    const cats = window.BLOCK_CATEGORIES;
    const kw = (filter || '').trim().toLowerCase();
    cats.forEach(cat => {
      const items = Object.values(window.BLOCK_DEFS).filter(d => d.category === cat.key);
      const matched = kw ? items.filter(d => d.name.toLowerCase().includes(kw) || d.type.toLowerCase().includes(kw)) : items;
      if (matched.length === 0) return;

      const section = document.createElement('div');
      section.className = 'tb-lib-section';
      const head = document.createElement('div');
      head.className = 'tb-lib-head';
      head.innerHTML = `<span class="tb-lib-caret">▾</span><span>${cat.label}</span><span class="tb-lib-count">${matched.length}</span>`;
      const list = document.createElement('div');
      list.className = 'tb-lib-list';
      matched.forEach(def => {
        const it = document.createElement('div');
        it.className = 'tb-lib-item';
        it.draggable = true;
        it.dataset.type = def.type;
        it.title = def.name;
        const delBtn = def._custom ? `<span class="tb-lib-del" data-del="${def.type}" title="删除自定义模块">✕</span>` : '';
        it.innerHTML = `<span class="tb-lib-icon">${def.icon}</span><span class="tb-lib-name">${def.name}</span>${delBtn}`;
        it.addEventListener('dragstart', (e) => {
          e.dataTransfer.setData('block-type', def.type);
          e.dataTransfer.effectAllowed = 'copy';
          it.classList.add('dragging');
        });
        it.addEventListener('dragend', () => { it.classList.remove('dragging'); });
        // 单击预览模块介绍
        it.addEventListener('click', (e) => {
          if (e.target.classList && e.target.classList.contains('tb-lib-del')) return;
          TB.selectLibType(def.type);
          // 高亮选中项
          host.querySelectorAll('.tb-lib-item').forEach(el => el.classList.remove('selected'));
          it.classList.add('selected');
        });
        // 双击直接添加到画布中心
        it.addEventListener('dblclick', () => {
          const vp = document.getElementById('tb-viewport');
          const r = vp.getBoundingClientRect();
          const w = window.Editor.screenToWorld(r.left + r.width / 2, r.top + r.height / 2);
          TB.addNode(def.type, w.x - 75, w.y - 32);
        });
        // 删除自定义模块
        const delEl = it.querySelector('.tb-lib-del');
        if (delEl) delEl.addEventListener('click', (e) => {
          e.stopPropagation();
          if (confirm('确定删除自定义模块「' + def.name + '」？')) {
            window.Custom.deleteCustomBlock(def.type);
            renderLibrary('');
            toast('已删除自定义模块', 'info');
          }
        });
        list.appendChild(it);
      });
      head.addEventListener('click', () => {
        section.classList.toggle('collapsed');
      });
      section.appendChild(head);
      section.appendChild(list);
      host.appendChild(section);
    });
    if (host.children.length === 0) {
      host.innerHTML = '<div class="tb-lib-empty">未找到匹配模块</div>';
    }
  }

  // ---------- 属性面板渲染 ----------
  function renderPropertyPanel() {
    const host = document.getElementById('tb-property');
    if (!host) return;

    // 优先级：多选 > 单选节点 > 侧边栏预览模块 > 模型设置
    const multiCount = (TB.state.selectedNodeIds || []).length;
    const nodeId = TB.state.selectedNodeId;
    const libType = TB.state.selectedLibType;

    if (multiCount > 0) {
      renderMultiSelectionPanel(host);
      return;
    }
    if (nodeId) {
      const node = TB.state.nodes.find(n => n.id === nodeId);
      if (node) {
        const def = window.BLOCK_DEFS[node.type];
        if (def) { renderNodeProperties(host, node, def); return; }
      }
    }
    if (libType) {
      const def = window.BLOCK_DEFS[libType];
      if (def) { renderLibPreview(host, def); return; }
    }
    renderModelSettings(host);
  }

  // 渲染模块介绍卡片（复用于节点选中与侧边栏预览）
  function renderDescCard(def) {
    const d = def.desc;
    if (!d) return '';
    return `<div class="tb-prop-desc">
      <div class="tb-desc-summary">${escapeHtml(d.summary)}</div>
      <div class="tb-desc-section">
        <div class="tb-desc-label">作用</div>
        <div class="tb-desc-text">${escapeHtml(d.role)}</div>
      </div>
      <div class="tb-desc-section">
        <div class="tb-desc-label">常用场景</div>
        <div class="tb-desc-text">${escapeHtml(d.uses)}</div>
      </div>
    </div>`;
  }

  // 侧边栏模块预览（仅介绍，无参数编辑）
  function renderLibPreview(host, def) {
    let html = `<div class="tb-prop-head">
      <span class="tb-prop-icon">${def.icon}</span>
      <div class="tb-prop-title"><div class="tb-prop-name">${def.name}</div><div class="tb-prop-type">${def.type}</div></div>
    </div>`;
    html += `<div class="tb-prop-body">`;
    html += renderDescCard(def);
    html += `<div class="tb-prop-empty">双击或拖拽该模块到画布即可使用</div>`;
    html += `</div>`;
    html += `<div class="tb-prop-actions">
      <button class="tb-btn tb-btn-primary" id="tb-lib-add">添加到画布</button>
    </div>`;
    host.innerHTML = html;
    const addBtn = host.querySelector('#tb-lib-add');
    if (addBtn) addBtn.addEventListener('click', () => {
      const vp = document.getElementById('tb-viewport');
      const r = vp.getBoundingClientRect();
      const w = window.Editor.screenToWorld(r.left + r.width / 2, r.top + r.height / 2);
      TB.addNode(def.type, w.x - 75, w.y - 32);
    });
  }

  // 选中节点：介绍 + 形状 + 参数
  function renderNodeProperties(host, node, def) {
    const shapes = TB.state.nodeShapes[node.id];
    const inShape = shapes && shapes.input && shapes.input[0];
    const outShape = shapes && shapes.output;
    const fmtS = (s) => !s || s.length === 0 ? '—' : '[' + s.map(v => v == null ? '?' : v).join(', ') + ']';

    let html = `<div class="tb-prop-head">
      <span class="tb-prop-icon">${def.icon}</span>
      <div class="tb-prop-title"><div class="tb-prop-name">${def.name}</div><div class="tb-prop-type">${node.type}</div></div>
    </div>`;
    html += `<div class="tb-prop-shapes">
      <div class="tb-prop-shape"><span>输入</span><code>${fmtS(inShape)}</code></div>
      <div class="tb-prop-shape"><span>输出</span><code>${fmtS(outShape)}</code></div>
    </div>`;
    html += `<div class="tb-prop-body" id="tb-prop-body"></div>`;
    html += `<div class="tb-prop-actions">
      <button class="tb-btn tb-btn-ghost" id="tb-prop-delete">删除节点</button>
      <button class="tb-btn tb-btn-ghost" id="tb-prop-dup">复制</button>
    </div>`;
    host.innerHTML = html;

    const body = host.querySelector('#tb-prop-body');
    // 介绍卡片
    if (def.desc) {
      const descWrap = document.createElement('div');
      descWrap.innerHTML = renderDescCard(def);
      while (descWrap.firstChild) body.appendChild(descWrap.firstChild);
    }
    // 参数表单
    if (def.paramSchema.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'tb-prop-empty';
      empty.textContent = '该模块无可配置参数';
      body.appendChild(empty);
    } else {
      def.paramSchema.forEach(field => {
        body.appendChild(buildField(node, field));
      });
    }

    host.querySelector('#tb-prop-delete').addEventListener('click', () => TB.removeNode(node.id));
    host.querySelector('#tb-prop-dup').addEventListener('click', () => {
      TB.addNode(node.type, node.x + 24, node.y + 24, deepClone(node.params));
    });
  }

  function buildField(node, field) {
    const wrap = document.createElement('div');
    wrap.className = 'tb-field';
    const val = node.params[field.key];
    const label = document.createElement('label');
    label.className = 'tb-field-label';
    label.textContent = field.label;
    wrap.appendChild(label);

    let input;
    if (field.type === 'select') {
      input = document.createElement('select');
      input.className = 'tb-input';
      field.options.forEach(opt => {
        const o = document.createElement('option');
        o.value = opt;
        o.textContent = opt;
        if (String(val) === String(opt)) o.selected = true;
        input.appendChild(o);
      });
      input.addEventListener('change', () => {
        TB.updateNodeParam(node.id, field.key, input.value, true);
      });
    } else if (field.type === 'text') {
      input = document.createElement('input');
      input.type = 'text';
      input.className = 'tb-input';
      input.value = val != null ? val : (field.default != null ? field.default : '');
      input.addEventListener('input', () => {
        TB.updateNodeParam(node.id, field.key, input.value, false);
      });
      input.addEventListener('change', () => { TB.pushHistory(); });
    } else {
      input = document.createElement('input');
      input.type = 'number';
      input.className = 'tb-input';
      if (field.min != null) input.min = field.min;
      if (field.max != null) input.max = field.max;
      if (field.step != null) input.step = field.step;
      input.value = val != null ? val : (field.default != null ? field.default : 0);
      input.addEventListener('input', () => {
        const v = input.value === '' ? 0 : Number(input.value);
        TB.updateNodeParam(node.id, field.key, v, false);
      });
      input.addEventListener('change', () => { TB.pushHistory(); });
    }
    wrap.appendChild(input);
    return wrap;
  }

  // ---------- 弹窗：打包自定义模块 ----------
  function showPackDialog() {
    const selCount = (TB.state.selectedNodeIds || []).length;
    if (selCount < 2) {
      toast('请按住 Shift 选中至少 2 个节点再打包', 'error');
      return;
    }
    const overlay = document.getElementById('tb-modal-overlay');
    const body = document.getElementById('tb-modal-body');
    if (!overlay || !body) return;

    body.innerHTML = `
      <div class="tb-modal-head">
        <h3>🧩 打包为自定义模块</h3>
      </div>
      <div class="tb-modal-content">
        <p class="tb-modal-tip">将选中的 <b>${selCount}</b> 个节点打包为一个可复用的自定义模块，自动处理维度推断与代码生成。</p>
        <div class="tb-field">
          <label class="tb-field-label">模块名称</label>
          <input id="tb-pack-name" class="tb-input" type="text" placeholder="如 MyBlock" value="MyBlock">
        </div>
        <div class="tb-field">
          <label class="tb-field-label">图标（emoji）</label>
          <input id="tb-pack-icon" class="tb-input" type="text" value="🧩" maxlength="2">
        </div>
        <div class="tb-field">
          <label class="tb-field-label">简介</label>
          <input id="tb-pack-summary" class="tb-input" type="text" placeholder="一句话描述" value="">
        </div>
        <div class="tb-field">
          <label class="tb-field-label">可配置参数（可选，每行一个，格式：key|标签|默认值）</label>
          <textarea id="tb-pack-params" class="tb-input tb-textarea" rows="3" placeholder="如：out_channels|输出通道|64"></textarea>
        </div>
      </div>
      <div class="tb-modal-actions">
        <button class="tb-btn tb-btn-ghost" id="tb-pack-cancel">取消</button>
        <button class="tb-btn tb-btn-primary" id="tb-pack-ok">创建模块</button>
      </div>
    `;
    overlay.classList.add('show');

    document.getElementById('tb-pack-cancel').addEventListener('click', closeModal);
    document.getElementById('tb-pack-ok').addEventListener('click', () => {
      const name = document.getElementById('tb-pack-name').value.trim() || 'MyBlock';
      const icon = document.getElementById('tb-pack-icon').value.trim() || '🧩';
      const summary = document.getElementById('tb-pack-summary').value.trim();
      const paramsText = document.getElementById('tb-pack-params').value.trim();

      // 解析参数 schema
      const paramSchema = [];
      const paramMapping = [];
      if (paramsText) {
        paramsText.split('\n').forEach(line => {
          const parts = line.split('|').map(s => s.trim());
          if (parts.length >= 3) {
            const key = parts[0], label = parts[1], def = parts[2];
            const numDef = Number(def);
            paramSchema.push({ key, label, type: isNaN(numDef) ? 'text' : 'number', default: isNaN(numDef) ? def : numDef });
            // 默认映射到子图中同名的节点参数（若存在）
            // 这里简化：不自动映射，用户可在子图中用同名参数
          }
        });
      }

      const desc = summary
        ? { summary, role: '由画布子图打包而来', uses: '可复用的自定义组合模块' }
        : undefined;

      closeModal();
      packCustomBlock({ name, icon, paramSchema, paramMapping, desc });
    });
  }

  // ---------- 弹窗：保存为模板 ----------
  function showSaveTemplateDialog() {
    if (TB.state.nodes.length === 0) { toast('画布为空，无法保存', 'error'); return; }
    const overlay = document.getElementById('tb-modal-overlay');
    const body = document.getElementById('tb-modal-body');
    if (!overlay || !body) return;

    body.innerHTML = `
      <div class="tb-modal-head">
        <h3>💾 保存为模板</h3>
      </div>
      <div class="tb-modal-content">
        <p class="tb-modal-tip">将当前画布的设计（${TB.state.nodes.length} 个节点）保存为可复用模板，下次可从模板菜单一键加载。</p>
        <div class="tb-field">
          <label class="tb-field-label">模板名称</label>
          <input id="tb-tpl-name" class="tb-input" type="text" placeholder="如 我的网络" value="${escapeHtml(TB.state.modelName)}">
        </div>
      </div>
      <div class="tb-modal-actions">
        <button class="tb-btn tb-btn-ghost" id="tb-tpl-cancel">取消</button>
        <button class="tb-btn tb-btn-primary" id="tb-tpl-ok">保存</button>
      </div>
    `;
    overlay.classList.add('show');

    document.getElementById('tb-tpl-cancel').addEventListener('click', closeModal);
    document.getElementById('tb-tpl-ok').addEventListener('click', () => {
      const name = document.getElementById('tb-tpl-name').value.trim() || 'MyTemplate';
      closeModal();
      saveCurrentAsTemplate(name);
    });
  }

  function closeModal() {
    const overlay = document.getElementById('tb-modal-overlay');
    if (overlay) overlay.classList.remove('show');
  }

  // ---------- AI 功能 ----------
  function showAiLoading(text) {
    const el = document.getElementById('tb-ai-loading');
    const txt = document.getElementById('tb-ai-loading-text');
    if (txt) txt.textContent = text || '正在分析…';
    if (el) el.classList.add('show');
  }
  function hideAiLoading() {
    const el = document.getElementById('tb-ai-loading');
    if (el) el.classList.remove('show');
  }

  // 设置 API Key 弹窗
  function showApiKeyDialog() {
    const overlay = document.getElementById('tb-modal-overlay');
    const body = document.getElementById('tb-modal-body');
    if (!overlay || !body) return;
    const currentKey = window.AI.getApiKey();
    const masked = currentKey ? currentKey.slice(0, 6) + '••••••' + currentKey.slice(-4) : '';
    body.innerHTML = `
      <div class="tb-modal-head"><h3>🔑 DeepSeek API Key 设置</h3></div>
      <div class="tb-modal-content">
        <p class="tb-modal-tip">输入你的 DeepSeek API Key，用于 AI 分析 PyTorch 代码。Key 保存在浏览器本地，不会上传。</p>
        <div class="tb-field">
          <label class="tb-field-label">API Key</label>
          <input id="tb-apikey-input" class="tb-input" type="password" placeholder="sk-..." value="${escapeHtml(currentKey)}">
        </div>
        <div class="tb-field">
          <label class="tb-field-label">当前状态</label>
          <div style="font-size:12px;color:var(--muted)">${currentKey ? '已设置：' + escapeHtml(masked) : '未设置'}</div>
        </div>
        <div class="tb-field">
          <label class="tb-field-label">获取方式</label>
          <div style="font-size:12px;color:var(--muted);line-height:1.6">
            访问 <span style="color:var(--accent)">platform.deepseek.com</span> 注册并创建 API Key。
            使用 deepseek-chat 模型，按 token 计费。
          </div>
        </div>
      </div>
      <div class="tb-modal-actions">
        <button class="tb-btn tb-btn-ghost" id="tb-apikey-cancel">取消</button>
        <button class="tb-btn tb-btn-primary" id="tb-apikey-save">保存</button>
      </div>
    `;
    overlay.classList.add('show');
    document.getElementById('tb-apikey-cancel').addEventListener('click', closeModal);
    document.getElementById('tb-apikey-save').addEventListener('click', () => {
      const key = document.getElementById('tb-apikey-input').value.trim();
      window.AI.setApiKey(key);
      closeModal();
      toast(key ? 'API Key 已保存' : 'API Key 已清除', 'success');
    });
  }

  // 粘贴代码分析弹窗
  function showPasteCodeDialog() {
    if (!window.AI.hasApiKey()) {
      toast('请先设置 DeepSeek API Key', 'error');
      showApiKeyDialog();
      return;
    }
    const overlay = document.getElementById('tb-modal-overlay');
    const body = document.getElementById('tb-modal-body');
    if (!overlay || !body) return;
    body.innerHTML = `
      <div class="tb-modal-head"><h3>📋 粘贴 PyTorch 代码</h3></div>
      <div class="tb-modal-content">
        <p class="tb-modal-tip">粘贴一段 PyTorch 模型代码（nn.Module 子类），AI 将分析其结构并转换为积木图。</p>
        <div class="tb-field">
          <label class="tb-field-label">代码</label>
          <textarea id="tb-paste-code" class="tb-textarea" rows="12" placeholder="import torch&#10;import torch.nn as nn&#10;&#10;class MyModel(nn.Module):&#10;    def __init__(self):&#10;        super().__init__()&#10;        ..."></textarea>
        </div>
      </div>
      <div class="tb-modal-actions">
        <button class="tb-btn tb-btn-ghost" id="tb-paste-cancel">取消</button>
        <button class="tb-btn tb-btn-primary" id="tb-paste-analyze">🤖 AI 分析</button>
      </div>
    `;
    overlay.classList.add('show');
    document.getElementById('tb-paste-cancel').addEventListener('click', closeModal);
    document.getElementById('tb-paste-analyze').addEventListener('click', () => {
      const code = document.getElementById('tb-paste-code').value.trim();
      if (!code) { toast('请粘贴代码', 'error'); return; }
      closeModal();
      runAiAnalysis(code);
    });
  }

  // 触发文件导入
  function triggerFileImport() {
    if (!window.AI.hasApiKey()) {
      toast('请先设置 DeepSeek API Key', 'error');
      showApiKeyDialog();
      return;
    }
    const input = document.getElementById('tb-file-input');
    if (!input) return;
    input.value = '';
    input.click();
  }

  // 处理文件选择
  function setupFileImport() {
    const input = document.getElementById('tb-file-input');
    if (!input) return;
    input.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const code = String(reader.result || '');
        if (!code.trim()) { toast('文件为空', 'error'); return; }
        toast('已导入：' + file.name, 'info');
        runAiAnalysis(code);
      };
      reader.onerror = () => toast('读取文件失败', 'error');
      reader.readAsText(file);
    });
  }

  // 执行 AI 分析
  async function runAiAnalysis(pyCode) {
    showAiLoading('正在发送代码到 DeepSeek 分析…');
    try {
      const result = await window.AI.analyzeCode(pyCode, (status) => {
        showAiLoading(status);
      });
      hideAiLoading();

      // 加载到画布
      TB.state.nodes = result.nodes;
      TB.state.edges = result.edges;
      autoLayout(TB.state.nodes, TB.state.edges);
      TB.state.selectedNodeId = null;
      TB.state.selectedEdgeId = null;
      TB.state.selectedNodeIds = [];
      TB.state.selectedLibType = null;
      if (result.modelName) TB.state.modelName = result.modelName;
      pushHistory();
      emit('graphChanged');
      emit('selectChanged');

      const msg = 'AI 分析完成：' + result.nodes.length + ' 个节点，' + result.edges.length + ' 条连线';
      if (result.warnings && result.warnings.length > 0) {
        toast(msg + '（' + result.warnings.length + ' 处警告）', 'info');
        console.log('[AI 分析警告]', result.warnings);
      } else {
        toast(msg, 'success');
      }
      if (window.Editor && window.Editor.fitView) setTimeout(() => window.Editor.fitView(), 100);
    } catch (err) {
      hideAiLoading();
      toast('AI 分析失败：' + (err.message || err), 'error');
      console.error('[AI 分析错误]', err);
    }
  }

  // ---------- 属性面板：多选状态 ----------
  function renderMultiSelectionPanel(host) {
    const count = (TB.state.selectedNodeIds || []).length;
    let html = `<div class="tb-prop-head">
      <span class="tb-prop-icon">🧩</span>
      <div class="tb-prop-title"><div class="tb-prop-name">多选模式</div><div class="tb-prop-type">${count} 个节点</div></div>
    </div>`;
    html += `<div class="tb-prop-body">
      <div class="tb-prop-desc">
        <div class="tb-desc-summary">已选中 ${count} 个节点</div>
        <div class="tb-desc-section">
          <div class="tb-desc-label">操作</div>
          <div class="tb-desc-text">可将这些节点打包为一个自定义模块，之后可从左侧「自定义模块」分类中拖拽复用。</div>
        </div>
      </div>
    </div>`;
    html += `<div class="tb-prop-actions">
      <button class="tb-btn tb-btn-primary" id="tb-multi-pack">🧩 打包为自定义模块</button>
      <button class="tb-btn tb-btn-ghost" id="tb-multi-clear">取消选择</button>
    </div>`;
    host.innerHTML = html;
    host.querySelector('#tb-multi-pack').addEventListener('click', showPackDialog);
    host.querySelector('#tb-multi-clear').addEventListener('click', () => {
      TB.state.selectedNodeIds = [];
      emit('selectChanged');
      if (window.Editor) window.Editor.render();
    });
  }

  function renderModelSettings(host) {
    const nodeCount = TB.state.nodes.length;
    const edgeCount = TB.state.edges.length;
    host.innerHTML = `
      <div class="tb-prop-head">
        <span class="tb-prop-icon">⚙️</span>
        <div class="tb-prop-title"><div class="tb-prop-name">模型设置</div><div class="tb-prop-type">Model</div></div>
      </div>
      <div class="tb-prop-body">
        <div class="tb-field">
          <label class="tb-field-label">类名</label>
          <input id="tb-model-name" class="tb-input" type="text" value="${escapeHtml(TB.state.modelName)}">
        </div>
        <div class="tb-field">
          <label class="tb-field-label">训练骨架</label>
          <label class="tb-switch">
            <input id="tb-train-toggle" type="checkbox" ${TB.state.includeTrainingLoop ? 'checked' : ''}>
            <span class="tb-switch-track"><span class="tb-switch-thumb"></span></span>
            <span class="tb-switch-text">${TB.state.includeTrainingLoop ? '已启用' : '未启用'}</span>
          </label>
        </div>
        <div class="tb-prop-stats">
          <div><span>节点</span><b>${nodeCount}</b></div>
          <div><span>连线</span><b>${edgeCount}</b></div>
        </div>
      </div>
      <div class="tb-prop-help">
        <div class="tb-help-title">使用说明</div>
        <ul>
          <li>从左侧拖拽模块到画布</li>
          <li>拖拽输出端口到输入端口连线</li>
          <li>点击节点编辑参数</li>
          <li>滚轮缩放，空白拖拽平移</li>
          <li>Delete 删除，Ctrl+D 复制</li>
          <li>Ctrl+Z / Ctrl+Y 撤销重做</li>
        </ul>
      </div>
    `;
    const nameInput = host.querySelector('#tb-model-name');
    nameInput.addEventListener('input', () => {
      TB.state.modelName = nameInput.value || 'MyModel';
      emit('graphChanged');
    });
    nameInput.addEventListener('change', () => TB.pushHistory());
    const toggle = host.querySelector('#tb-train-toggle');
    toggle.addEventListener('change', () => {
      TB.state.includeTrainingLoop = toggle.checked;
      host.querySelector('.tb-switch-text').textContent = toggle.checked ? '已启用' : '未启用';
      TB.pushHistory();
      emit('graphChanged');
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // ---------- 代码预览 ----------
  function highlight(code) {
    const kw = new Set(['import', 'as', 'class', 'def', 'return', 'self', 'if', 'for', 'in', 'while',
      'True', 'False', 'None', 'super', 'lambda', 'with', 'try', 'except', 'pass', 'break', 'continue',
      'and', 'or', 'not', 'is', 'yield', 'global', 'nonlocal', 'elif', 'else', 'assert', 'raise']);
    const types = new Set(['nn', 'torch', 'F']);
    const re = /(#[^\n]*)|('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|(\b\d+\.?\d*\b)|(\b[A-Za-z_]\w*\b)/g;
    const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    let out = '', last = 0, m;
    while ((m = re.exec(code)) !== null) {
      out += esc(code.slice(last, m.index));
      const [full, comment, str, num, ident] = m;
      if (comment) out += `<span class="c">${esc(comment)}</span>`;
      else if (str) out += `<span class="s">${esc(str)}</span>`;
      else if (num) out += `<span class="n">${esc(num)}</span>`;
      else if (ident) {
        if (kw.has(ident)) out += `<span class="k">${esc(ident)}</span>`;
        else if (types.has(ident)) out += `<span class="t">${esc(ident)}</span>`;
        else out += esc(ident);
      }
      last = m.index + full.length;
    }
    out += esc(code.slice(last));
    return out;
  }

  function updateCodePreview() {
    const code = window.CodeGen.generate(TB.state.nodes, TB.state.edges, {
      modelName: TB.state.modelName,
      includeTrainingLoop: TB.state.includeTrainingLoop
    }).code;
    TB._lastCode = code;
    const pre = document.getElementById('tb-code-pre');
    if (pre) pre.innerHTML = highlight(code) || '<span class="c"># 暂无代码</span>';
    const empty = document.getElementById('tb-code-empty');
    if (empty) empty.style.display = TB.state.nodes.length === 0 ? 'flex' : 'none';
  }

  // ---------- 错误计数 ----------
  function updateErrorBadge() {
    const badge = document.getElementById('tb-error-badge');
    if (!badge) return;
    const n = TB.state.shapeErrors.length;
    badge.textContent = n;
    badge.style.display = n > 0 ? 'inline-flex' : 'none';
    badge.title = n > 0 ? `${n} 个形状错误` : '无错误';
  }

  function updateUndoRedo() {
    const undoBtn = document.getElementById('tb-undo');
    const redoBtn = document.getElementById('tb-redo');
    if (undoBtn) undoBtn.disabled = TB.state.historyIndex <= 0;
    if (redoBtn) redoBtn.disabled = TB.state.historyIndex >= TB.state.history.length - 1;
  }

  TB.updateZoomLabel = function () {
    const el = document.getElementById('tb-zoom-label');
    if (el) el.textContent = Math.round(TB.state.zoom * 100) + '%';
  };

  // ---------- graphChanged 处理（防抖 300ms） ----------
  const onGraphChanged = debounce(() => {
    const res = window.ShapeInference.infer(TB.state.nodes, TB.state.edges);
    TB.state.nodeShapes = res.nodeShapes;
    TB.state.shapeErrors = res.errors;
    if (window.Editor) window.Editor.render();
    updateErrorBadge();
    updateCodePreview();
    const hint = document.getElementById('tb-canvas-hint');
    if (hint) hint.style.display = TB.state.nodes.length === 0 ? 'block' : 'none';
  }, 300);

  // ---------- 模板 ----------
  const TEMPLATES = [
    {
      key: 'lenet', name: 'LeNet-5',
      modelName: 'LeNet5',
      nodes: [
        { id: 'n1', type: 'input', x: 40, y: 60, params: { shape: '[1, 1, 32, 32]' } },
        { id: 'n2', type: 'conv2d', x: 260, y: 60, params: { in_channels: 1, out_channels: 6, kernel_size: 5, stride: 1, padding: 0 } },
        { id: 'n3', type: 'maxpool2d', x: 480, y: 60, params: { kernel_size: 2, stride: 2 } },
        { id: 'n4', type: 'conv2d', x: 700, y: 60, params: { in_channels: 6, out_channels: 16, kernel_size: 5, stride: 1, padding: 0 } },
        { id: 'n5', type: 'maxpool2d', x: 920, y: 60, params: { kernel_size: 2, stride: 2 } },
        { id: 'n6', type: 'flatten', x: 480, y: 200, params: { start_dim: 1, end_dim: -1 } },
        { id: 'n7', type: 'linear', x: 700, y: 200, params: { in_features: 400, out_features: 120 } },
        { id: 'n8', type: 'linear', x: 920, y: 200, params: { in_features: 120, out_features: 84 } },
        { id: 'n9', type: 'linear', x: 1140, y: 200, params: { in_features: 84, out_features: 10 } },
        { id: 'n10', type: 'output', x: 1140, y: 60, params: {} }
      ],
      edges: [
        { id: 'e1', from: { nodeId: 'n1', port: 0 }, to: { nodeId: 'n2', port: 0 } },
        { id: 'e2', from: { nodeId: 'n2', port: 0 }, to: { nodeId: 'n3', port: 0 } },
        { id: 'e3', from: { nodeId: 'n3', port: 0 }, to: { nodeId: 'n4', port: 0 } },
        { id: 'e4', from: { nodeId: 'n4', port: 0 }, to: { nodeId: 'n5', port: 0 } },
        { id: 'e5', from: { nodeId: 'n5', port: 0 }, to: { nodeId: 'n6', port: 0 } },
        { id: 'e6', from: { nodeId: 'n6', port: 0 }, to: { nodeId: 'n7', port: 0 } },
        { id: 'e7', from: { nodeId: 'n7', port: 0 }, to: { nodeId: 'n8', port: 0 } },
        { id: 'e8', from: { nodeId: 'n8', port: 0 }, to: { nodeId: 'n9', port: 0 } },
        { id: 'e9', from: { nodeId: 'n9', port: 0 }, to: { nodeId: 'n10', port: 0 } }
      ]
    },
    {
      key: 'resnet', name: 'ResNet 残差块',
      modelName: 'ResidualBlock',
      nodes: [
        { id: 'n1', type: 'input', x: 40, y: 120, params: { shape: '[1, 64, 56, 56]' } },
        { id: 'n2', type: 'conv2d', x: 260, y: 60, params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 } },
        { id: 'n3', type: 'batchnorm2d', x: 480, y: 60, params: { num_features: 64 } },
        { id: 'n4', type: 'relu', x: 700, y: 60, params: {} },
        { id: 'n5', type: 'conv2d', x: 920, y: 60, params: { in_channels: 64, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 } },
        { id: 'n6', type: 'batchnorm2d', x: 1140, y: 60, params: { num_features: 64 } },
        { id: 'n7', type: 'add', x: 1140, y: 220, params: {} },
        { id: 'n8', type: 'relu', x: 1360, y: 220, params: {} },
        { id: 'n9', type: 'output', x: 1580, y: 220, params: {} }
      ],
      edges: [
        { id: 'e1', from: { nodeId: 'n1', port: 0 }, to: { nodeId: 'n2', port: 0 } },
        { id: 'e2', from: { nodeId: 'n2', port: 0 }, to: { nodeId: 'n3', port: 0 } },
        { id: 'e3', from: { nodeId: 'n3', port: 0 }, to: { nodeId: 'n4', port: 0 } },
        { id: 'e4', from: { nodeId: 'n4', port: 0 }, to: { nodeId: 'n5', port: 0 } },
        { id: 'e5', from: { nodeId: 'n5', port: 0 }, to: { nodeId: 'n6', port: 0 } },
        { id: 'e6', from: { nodeId: 'n6', port: 0 }, to: { nodeId: 'n7', port: 1 } },
        { id: 'e7', from: { nodeId: 'n1', port: 0 }, to: { nodeId: 'n7', port: 0 } },
        { id: 'e8', from: { nodeId: 'n7', port: 0 }, to: { nodeId: 'n8', port: 0 } },
        { id: 'e9', from: { nodeId: 'n8', port: 0 }, to: { nodeId: 'n9', port: 0 } }
      ]
    },
    {
      key: 'transformer', name: 'Transformer Encoder',
      modelName: 'TransformerEncoder',
      nodes: [
        { id: 'n1', type: 'input', x: 40, y: 140, params: { shape: '[8, 16, 64]' } },
        { id: 'n2', type: 'layernorm', x: 260, y: 140, params: { normalized_shape: 64 } },
        { id: 'n3', type: 'multiheadattention', x: 480, y: 140, params: { embed_dim: 64, num_heads: 4, batch_first: 'true' } },
        { id: 'n4', type: 'add', x: 700, y: 140, params: {} },
        { id: 'n5', type: 'layernorm', x: 920, y: 140, params: { normalized_shape: 64 } },
        { id: 'n6', type: 'linear', x: 1140, y: 80, params: { in_features: 64, out_features: 256 } },
        { id: 'n7', type: 'gelu', x: 1360, y: 80, params: {} },
        { id: 'n8', type: 'linear', x: 1360, y: 200, params: { in_features: 256, out_features: 64 } },
        { id: 'n9', type: 'add', x: 1580, y: 140, params: {} },
        { id: 'n10', type: 'output', x: 1800, y: 140, params: {} }
      ],
      edges: [
        { id: 'e1', from: { nodeId: 'n1', port: 0 }, to: { nodeId: 'n2', port: 0 } },
        { id: 'e2', from: { nodeId: 'n2', port: 0 }, to: { nodeId: 'n3', port: 0 } },
        { id: 'e3', from: { nodeId: 'n3', port: 0 }, to: { nodeId: 'n4', port: 1 } },
        { id: 'e4', from: { nodeId: 'n1', port: 0 }, to: { nodeId: 'n4', port: 0 } },
        { id: 'e5', from: { nodeId: 'n4', port: 0 }, to: { nodeId: 'n5', port: 0 } },
        { id: 'e6', from: { nodeId: 'n5', port: 0 }, to: { nodeId: 'n6', port: 0 } },
        { id: 'e7', from: { nodeId: 'n6', port: 0 }, to: { nodeId: 'n7', port: 0 } },
        { id: 'e8', from: { nodeId: 'n7', port: 0 }, to: { nodeId: 'n8', port: 0 } },
        { id: 'e9', from: { nodeId: 'n8', port: 0 }, to: { nodeId: 'n9', port: 1 } },
        { id: 'e10', from: { nodeId: 'n4', port: 0 }, to: { nodeId: 'n9', port: 0 } },
        { id: 'e11', from: { nodeId: 'n9', port: 0 }, to: { nodeId: 'n10', port: 0 } }
      ]
    }
  ];

  // ---------- 导出 ----------
  function download(filename, content, mime) {
    const blob = new Blob([content], { type: mime || 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 100);
  }

  function exportPython() {
    const code = TB._lastCode || window.CodeGen.generate(TB.state.nodes, TB.state.edges, {
      modelName: TB.state.modelName, includeTrainingLoop: TB.state.includeTrainingLoop
    }).code;
    download((TB.state.modelName || 'model') + '.py', code, 'text/x-python');
    toast('已导出 Python 脚本', 'success');
  }

  function exportNotebook() {
    const code = TB._lastCode || window.CodeGen.generate(TB.state.nodes, TB.state.edges, {
      modelName: TB.state.modelName, includeTrainingLoop: false
    }).code;
    const nb = {
      cells: [
        { cell_type: 'markdown', metadata: {}, source: ['# ' + (TB.state.modelName || 'MyModel') + '\n', '\n', '由 TorchBlox 自动生成。'] },
        { cell_type: 'code', execution_count: null, metadata: {}, outputs: [], source: code.split('\n').map(l => l + '\n') }
      ],
      metadata: { kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' }, language_info: { name: 'python' } },
      nbformat: 4, nbformat_minor: 5
    };
    download((TB.state.modelName || 'model') + '.ipynb', JSON.stringify(nb, null, 2), 'application/json');
    toast('已导出 Jupyter Notebook', 'success');
  }

  function exportOnnxScript() {
    const name = TB.state.modelName || 'MyModel';
    const code = TB._lastCode || window.CodeGen.generate(TB.state.nodes, TB.state.edges, { modelName: name }).code;
    const script = code + '\n\n\n' +
      '# ===== ONNX 导出 =====\n' +
      'import torch\n\n' +
      `model = ${name}()\n` +
      'model.eval()\n' +
      'dummy = torch.randn(1, 3, 224, 224)  # 请按实际输入形状修改\n' +
      `torch.onnx.export(model, dummy, "${name.toLowerCase()}.onnx", input_names=["input"], output_names=["output"], dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})\n` +
      `print("已导出 ${name.toLowerCase()}.onnx")\n`;
    download(name.toLowerCase() + '_onnx.py', script, 'text/x-python');
    toast('已导出 ONNX 导出脚本', 'success');
  }

  function exportTorchScript() {
    const name = TB.state.modelName || 'MyModel';
    const code = TB._lastCode || window.CodeGen.generate(TB.state.nodes, TB.state.edges, { modelName: name }).code;
    const script = code + '\n\n\n' +
      '# ===== TorchScript 导出 =====\n' +
      'import torch\n\n' +
      `model = ${name}()\n` +
      'model.eval()\n' +
      'dummy = torch.randn(1, 3, 224, 224)  # 请按实际输入形状修改\n' +
      `scripted = torch.jit.trace(model, dummy)\n` +
      `scripted.save("${name.toLowerCase()}.pt")\n` +
      `print("已导出 ${name.toLowerCase()}.pt")\n`;
    download(name.toLowerCase() + '_torchscript.py', script, 'text/x-python');
    toast('已导出 TorchScript 脚本', 'success');
  }

  // ---------- 下拉菜单 ----------
  function setupDropdowns() {
    // 模板下拉：用 refreshTemplateMenu 初始化（含用户模板）
    const tplBtn = document.getElementById('tb-tpl-btn');
    const tplMenu = document.getElementById('tb-tpl-menu');
    refreshTemplateMenu();

    // 导出下拉
    const expBtn = document.getElementById('tb-export-btn');
    const expMenu = document.getElementById('tb-export-menu');
    if (expMenu) {
      expMenu.querySelector('[data-export="python"]').addEventListener('click', () => { exportPython(); closeAllMenus(); });
      expMenu.querySelector('[data-export="notebook"]').addEventListener('click', () => { exportNotebook(); closeAllMenus(); });
      expMenu.querySelector('[data-export="onnx"]').addEventListener('click', () => { exportOnnxScript(); closeAllMenus(); });
      expMenu.querySelector('[data-export="torchscript"]').addEventListener('click', () => { exportTorchScript(); closeAllMenus(); });
    }

    // AI 下拉
    const aiBtn = document.getElementById('tb-ai-btn');
    const aiMenu = document.getElementById('tb-ai-menu');
    if (aiMenu) {
      const importItem = aiMenu.querySelector('#tb-ai-import');
      const pasteItem = aiMenu.querySelector('#tb-ai-paste');
      const keyItem = aiMenu.querySelector('#tb-ai-key');
      if (importItem) importItem.addEventListener('click', () => { closeAllMenus(); triggerFileImport(); });
      if (pasteItem) pasteItem.addEventListener('click', () => { closeAllMenus(); showPasteCodeDialog(); });
      if (keyItem) keyItem.addEventListener('click', () => { closeAllMenus(); showApiKeyDialog(); });
    }

    // 下拉按钮点击切换
    [
      [tplBtn, tplMenu],
      [expBtn, expMenu],
      [aiBtn, aiMenu]
    ].forEach(([btn, menu]) => {
      if (!btn || !menu) return;
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const open = menu.classList.contains('show');
        closeAllMenus();
        if (!open) menu.classList.add('show');
      });
    });

    document.addEventListener('click', closeAllMenus);

    // 模态框点击遮罩关闭 + ESC 关闭
    const modalOverlay = document.getElementById('tb-modal-overlay');
    if (modalOverlay) {
      modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
      });
    }
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        const ov = document.getElementById('tb-modal-overlay');
        if (ov && ov.classList.contains('show')) {
          closeModal();
          e.stopPropagation();
        }
      }
    });
  }
  function closeAllMenus() {
    document.querySelectorAll('.tb-menu.show').forEach(m => m.classList.remove('show'));
  }

  // ---------- 工具栏按钮 ----------
  function setupToolbar() {
    document.getElementById('tb-undo').addEventListener('click', () => TB.undo());
    document.getElementById('tb-redo').addEventListener('click', () => TB.redo());
    document.getElementById('tb-zoom-out').addEventListener('click', () => window.Editor.zoomBy(1 / 1.2));
    document.getElementById('tb-zoom-in').addEventListener('click', () => window.Editor.zoomBy(1.2));
    document.getElementById('tb-zoom-reset').addEventListener('click', () => window.Editor.zoomReset());
    document.getElementById('tb-fit').addEventListener('click', () => window.Editor.fitView());

    // 打包自定义模块按钮
    const packBtn = document.getElementById('tb-pack-btn');
    if (packBtn) packBtn.addEventListener('click', () => {
      // 若没有多选，提示用户先 Shift+点击选择
      const cnt = (TB.state.selectedNodeIds || []).length;
      if (cnt < 2) {
        toast('请先按住 Shift 点击选中至少 2 个节点再打包', 'info');
        return;
      }
      showPackDialog();
    });

    // 侧栏折叠
    const toggleLib = document.getElementById('tb-toggle-lib');
    const toggleProp = document.getElementById('tb-toggle-prop');
    const toggleCode = document.getElementById('tb-toggle-code');
    if (toggleLib) toggleLib.addEventListener('click', () => {
      document.body.classList.toggle('lib-collapsed');
    });
    if (toggleProp) toggleProp.addEventListener('click', () => {
      document.body.classList.toggle('prop-collapsed');
    });
    if (toggleCode) toggleCode.addEventListener('click', () => {
      document.body.classList.toggle('code-collapsed');
    });

    // 复制代码
    const copyBtn = document.getElementById('tb-copy-code');
    if (copyBtn) copyBtn.addEventListener('click', () => {
      const code = TB._lastCode || '';
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(code).then(() => toast('代码已复制', 'success'), () => fallbackCopy(code));
      } else fallbackCopy(code);
    });
  }
  function fallbackCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); toast('代码已复制', 'success'); }
    catch (e) { toast('复制失败', 'error'); }
    document.body.removeChild(ta);
  }

  // ---------- 初始化 ----------
  function init() {
    window.TB = TB;
    // 初始空历史快照
    TB.state.history.push(snapshot());
    TB.state.historyIndex = 0;

    // 加载 localStorage 中的自定义模块并注册到 BLOCK_DEFS
    if (window.Custom && window.Custom.loadAndRegisterAll) {
      window.Custom.loadAndRegisterAll();
    }

    renderLibrary('');
    renderPropertyPanel();
    setupDropdowns();
    setupToolbar();
    setupFileImport();
    updateUndoRedo();
    updateErrorBadge();
    updateCodePreview();
    const hint0 = document.getElementById('tb-canvas-hint');
    if (hint0) hint0.style.display = TB.state.nodes.length === 0 ? 'block' : 'none';

    // 搜索
    const search = document.getElementById('tb-search');
    if (search) search.addEventListener('input', () => renderLibrary(search.value));

    // 事件
    on('graphChanged', onGraphChanged);
    on('selectChanged', () => {
      renderPropertyPanel();
      if (window.Editor) window.Editor.render();
    });

    // 初始化编辑器
    if (window.Editor && window.Editor.init) window.Editor.init();
    if (window.Editor) window.Editor.render();
    TB.updateZoomLabel();

    // 默认加载 LeNet 模板，展示效果
    setTimeout(() => {
      TB.loadTemplate(TEMPLATES[0]);
    }, 100);
  }

  // DOM 就绪后初始化
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.TB = TB;
})();

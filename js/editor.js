/* =====================================================================
 * TorchBlox — editor.js
 * 画布交互引擎：SVG(连线) + DOM(节点) 混合画布，共用一个 transform。
 * 负责：渲染节点/连线、侧栏拖入、节点移动、端口连线、平移缩放、
 *       选中、删除、复制、撤销/重做快捷键。
 * ===================================================================== */
(function () {
  'use strict';

  const Editor = {
    viewport: null,
    world: null,
    nodeLayer: null,
    edgeLayer: null,   // <svg>
    edgeGroup: null,   // <g> inside svg
    tempEdge: null,
    // 交互状态
    drag: null,        // { nodeId, offX, offY, moved }
    pan: null,         // { active, startX, startY, origX, origY, moved }
    connect: null      // { fromNode, fromPort, mouseX, mouseY }
  };

  // ---------- 坐标换算 ----------
  function screenToWorld(clientX, clientY) {
    const r = Editor.world.getBoundingClientRect();
    const zoom = window.TB.state.zoom;
    return {
      x: (clientX - r.left) / zoom,
      y: (clientY - r.top) / zoom
    };
  }

  function applyTransform() {
    const { pan, zoom } = window.TB.state;
    Editor.world.style.transform = `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`;
  }

  // ---------- 端口坐标（世界坐标） ----------
  function getPortWorld(nodeId, portIndex, dir) {
    const sel = `.tb-port[data-node-id="${cssEsc(nodeId)}"][data-port="${portIndex}"][data-dir="${dir}"]`;
    const el = Editor.nodeLayer.querySelector(sel);
    if (!el) return null;
    const pr = el.getBoundingClientRect();
    const wr = Editor.world.getBoundingClientRect();
    const zoom = window.TB.state.zoom;
    return {
      x: (pr.left + pr.width / 2 - wr.left) / zoom,
      y: (pr.top + pr.height / 2 - wr.top) / zoom
    };
  }

  function cssEsc(s) {
    return String(s).replace(/[^a-zA-Z0-9_-]/g, m => '\\' + m);
  }

  // ---------- 形状标签文本 ----------
  function fmtShape(s) {
    if (!s || s.length === 0) return '—';
    return '[' + s.map(v => (v == null ? '?' : v)).join(', ') + ']';
  }

  // ---------- 渲染 ----------
  function render() {
    renderNodes();
    renderEdges();
  }

  function renderNodes() {
    const layer = Editor.nodeLayer;
    layer.innerHTML = '';
    const { nodes, selectedNodeId, selectedNodeIds, nodeShapes, shapeErrors } = window.TB.state;
    const errNodeIds = new Set();
    (shapeErrors || []).forEach(e => { if (e.nodeId) errNodeIds.add(e.nodeId); });
    const multiSel = new Set(selectedNodeIds || []);

    nodes.forEach(node => {
      const def = window.BLOCK_DEFS[node.type];
      if (!def) return;
      const el = document.createElement('div');
      el.className = 'tb-node' + (selectedNodeId === node.id ? ' selected' : '');
      if (multiSel.has(node.id)) el.classList.add('multi-selected');
      if (errNodeIds.has(node.id)) el.classList.add('has-error');
      el.dataset.id = node.id;
      el.style.left = node.x + 'px';
      el.style.top = node.y + 'px';

      // 形状标签
      const shapes = nodeShapes && nodeShapes[node.id];
      let outShape = shapes ? shapes.output : null;
      if ((!outShape || outShape.length === 0) && shapes && shapes.input && shapes.input.length) {
        outShape = shapes.input[0];
      }
      const shapeText = fmtShape(outShape);
      const hasErr = errNodeIds.has(node.id);

      // 端口
      let portsHtml = '';
      for (let i = 0; i < def.inputs; i++) {
        const pct = ((i + 1) / (def.inputs + 1)) * 100;
        portsHtml += `<span class="tb-port in" data-node-id="${node.id}" data-port="${i}" data-dir="in" style="top:${pct}%" title="输入 ${i}"></span>`;
      }
      for (let i = 0; i < def.outputs; i++) {
        const pct = ((i + 1) / (def.outputs + 1)) * 100;
        portsHtml += `<span class="tb-port out" data-node-id="${node.id}" data-port="${i}" data-dir="out" style="top:${pct}%" title="输出 ${i}"></span>`;
      }

      el.innerHTML =
        `<div class="tb-node-head" data-drag="1">
          <span class="tb-node-icon">${def.icon}</span>
          <span class="tb-node-name">${def.name}</span>
        </div>
        <div class="tb-node-shape${hasErr ? ' err' : ''}">${hasErr ? '⚠ ' : ''}${shapeText}</div>
        ${portsHtml}`;
      layer.appendChild(el);
    });
  }

  function renderEdges() {
    const g = Editor.edgeGroup;
    g.innerHTML = '';
    const { edges, selectedEdgeId, shapeErrors } = window.TB.state;
    const errEdgeIds = new Set();
    (shapeErrors || []).forEach(e => { if (e.edgeId) errEdgeIds.add(e.edgeId); });

    edges.forEach(edge => {
      const from = getPortWorld(edge.from.nodeId, edge.from.port, 'out');
      const to = getPortWorld(edge.to.nodeId, edge.to.port, 'in');
      if (!from || !to) return;
      const d = bezierPath(from, to);
      const isErr = errEdgeIds.has(edge.id);
      const isSel = selectedEdgeId === edge.id;
      const color = isErr ? '#ef4444' : (isSel ? '#f59e0b' : '#3b82f6');

      // 透明宽命中区
      const hit = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      hit.setAttribute('d', d);
      hit.setAttribute('class', 'tb-edge-hit');
      hit.setAttribute('stroke', 'transparent');
      hit.setAttribute('stroke-width', '14');
      hit.setAttribute('fill', 'none');
      hit.dataset.edgeId = edge.id;
      g.appendChild(hit);

      // 可见线
      const vis = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      vis.setAttribute('d', d);
      vis.setAttribute('stroke', color);
      vis.setAttribute('stroke-width', isSel ? '2.5' : '2');
      vis.setAttribute('fill', 'none');
      vis.setAttribute('class', 'tb-edge' + (isErr ? ' err' : ''));
      vis.setAttribute('marker-end', 'url(#tb-arrow)');
      g.appendChild(vis);
    });
  }

  function bezierPath(a, b) {
    const dx = Math.max(40, Math.abs(b.x - a.x) * 0.5);
    return `M ${a.x},${a.y} C ${a.x + dx},${a.y} ${b.x - dx},${b.y} ${b.x},${b.y}`;
  }

  // 仅重绘连线（节点拖拽时调用，避免重建节点 DOM）
  function updateEdges() {
    renderEdges();
  }

  // ---------- 事件：侧栏拖入 ----------
  function setupDrop() {
    const vp = Editor.viewport;
    vp.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });
    vp.addEventListener('drop', (e) => {
      e.preventDefault();
      const type = e.dataTransfer.getData('block-type');
      if (!type || !window.BLOCK_DEFS[type]) return;
      const w = screenToWorld(e.clientX, e.clientY);
      const def = window.BLOCK_DEFS[type];
      // 居中放置
      window.TB.addNode(type, w.x - 75, w.y - 32);
    });
  }

  // ---------- 事件：画布交互（统一在 viewport 上委托） ----------
  function setupCanvas() {
    const vp = Editor.viewport;

    vp.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    vp.addEventListener('wheel', onWheel, { passive: false });
    vp.addEventListener('click', onClick);
  }

  function onMouseDown(e) {
    if (e.button !== 0) return;
    const target = e.target;

    // 1. 输出端口 → 开始连线
    if (target.classList && target.classList.contains('tb-port') && target.dataset.dir === 'out') {
      startConnect(target, e);
      e.preventDefault();
      return;
    }

    // 2. 连线命中区 → 选中连线
    if (target.classList && target.classList.contains('tb-edge-hit')) {
      window.TB.selectEdge(target.dataset.edgeId);
      e.preventDefault();
      return;
    }

    // 3. 节点 → 选中；标题栏则同时开始拖拽
    const nodeEl = target.closest && target.closest('.tb-node');
    if (nodeEl) {
      const id = nodeEl.dataset.id;
      const node = window.TB.state.nodes.find(n => n.id === id);
      if (node) {
        // Shift+点击：多选切换
        if (e.shiftKey) {
          const arr = window.TB.state.selectedNodeIds || [];
          const idx = arr.indexOf(id);
          if (idx >= 0) {
            window.TB.state.selectedNodeIds = arr.filter(x => x !== id);
          } else {
            window.TB.state.selectedNodeIds = arr.concat([id]);
          }
          // 多选时不单独选中，但仍触发渲染
          window.TB.state.selectedNodeId = null;
          window.TB.emit('selectChanged');
          if (window.Editor) window.Editor.render();
          return;
        }
        window.TB.selectNode(id);
        // 若已有选中集合且未按 shift，清空多选
        if (window.TB.state.selectedNodeIds && window.TB.state.selectedNodeIds.length) {
          window.TB.state.selectedNodeIds = [];
          if (window.Editor) window.Editor.render();
        }
        const head = target.closest('.tb-node-head');
        if (head) {
          const w = screenToWorld(e.clientX, e.clientY);
          Editor.drag = { nodeId: id, offX: w.x - node.x, offY: w.y - node.y, moved: false };
          e.preventDefault();
        }
      }
      return;
    }

    // 4. 空白 → 平移
    Editor.pan = {
      active: true, startX: e.clientX, startY: e.clientY,
      origX: window.TB.state.pan.x, origY: window.TB.state.pan.y, moved: false
    };
    window.TB.selectNode(null);
    window.TB.selectEdge(null);
    // 清除侧边栏选中高亮
    if (window.TB.state.selectedLibType) window.TB.selectLibType(null);
    document.querySelectorAll('.tb-lib-item.selected').forEach(el => el.classList.remove('selected'));
    Editor.viewport.classList.add('panning');
  }

  function onMouseMove(e) {
    if (Editor.drag) {
      const w = screenToWorld(e.clientX, e.clientY);
      const node = window.TB.state.nodes.find(n => n.id === Editor.drag.nodeId);
      if (node) {
        node.x = Math.round(w.x - Editor.drag.offX);
        node.y = Math.round(w.y - Editor.drag.offY);
        Editor.drag.moved = true;
        // 直接更新 DOM 位置，避免重建
        const el = Editor.nodeLayer.querySelector(`.tb-node[data-id="${cssEsc(node.id)}"]`);
        if (el) { el.style.left = node.x + 'px'; el.style.top = node.y + 'px'; }
        updateEdges();
      }
      return;
    }
    if (Editor.pan && Editor.pan.active) {
      const dx = e.clientX - Editor.pan.startX;
      const dy = e.clientY - Editor.pan.startY;
      if (Math.abs(dx) + Math.abs(dy) > 3) Editor.pan.moved = true;
      window.TB.state.pan.x = Editor.pan.origX + dx;
      window.TB.state.pan.y = Editor.pan.origY + dy;
      applyTransform();
      return;
    }
    if (Editor.connect && Editor.connect.active) {
      const w = screenToWorld(e.clientX, e.clientY);
      Editor.connect.mouseX = w.x;
      Editor.connect.mouseY = w.y;
      updateTempEdge();
    }
  }

  function onMouseUp(e) {
    if (Editor.drag) {
      if (Editor.drag.moved) {
        window.TB.pushHistory();
        window.TB.emit('graphChanged');
      }
      Editor.drag = null;
    }
    if (Editor.pan) {
      Editor.viewport.classList.remove('panning');
      Editor.pan = null;
    }
    if (Editor.connect && Editor.connect.active) {
      finishConnect(e);
    }
  }

  function onClick(e) {
    // 点击空白时取消选中已在 mousedown 处理；此处处理连线/节点点击的选中已由 mousedown 完成
    // pan 后的 click 屏蔽
    if (Editor.pan && Editor.pan.moved) {
      e.stopPropagation();
      e.preventDefault();
    }
  }

  // ---------- 连线 ----------
  function startConnect(portEl, e) {
    Editor.connect = {
      active: true,
      fromNode: portEl.dataset.nodeId,
      fromPort: parseInt(portEl.dataset.port, 10),
      mouseX: 0, mouseY: 0
    };
    const w = screenToWorld(e.clientX, e.clientY);
    Editor.connect.mouseX = w.x;
    Editor.connect.mouseY = w.y;
    // 创建临时 path
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('class', 'tb-edge-temp');
    p.setAttribute('stroke', '#f59e0b');
    p.setAttribute('stroke-width', '2');
    p.setAttribute('stroke-dasharray', '5,4');
    p.setAttribute('fill', 'none');
    Editor.tempEdge = p;
    Editor.edgeGroup.appendChild(p);
    updateTempEdge();
  }

  function updateTempEdge() {
    if (!Editor.connect || !Editor.tempEdge) return;
    const from = getPortWorld(Editor.connect.fromNode, Editor.connect.fromPort, 'out');
    if (!from) return;
    const to = { x: Editor.connect.mouseX, y: Editor.connect.mouseY };
    Editor.tempEdge.setAttribute('d', bezierPath(from, to));
  }

  function finishConnect(e) {
    const el = document.elementFromPoint(e.clientX, e.clientY);
    let target = el;
    if (el && !el.classList.contains('tb-port')) {
      // elementFromPoint 可能命中端口的伪元素或子节点
    }
    if (target && target.classList && target.classList.contains('tb-port') && target.dataset.dir === 'in') {
      const toNode = target.dataset.nodeId;
      const toPort = parseInt(target.dataset.port, 10);
      const fromNode = Editor.connect.fromNode;
      const fromPort = Editor.connect.fromPort;
      // 校验
      if (toNode === fromNode) {
        flashError('不能连接到自身');
      } else {
        // 一个输入端口只能连一条边：移除已有
        const existing = window.TB.state.edges.find(ed =>
          ed.to.nodeId === toNode && ed.to.port === toPort);
        if (existing) window.TB.removeEdge(existing.id, true);
        window.TB.addEdge({ nodeId: fromNode, port: fromPort }, { nodeId: toNode, port: toPort });
      }
    }
    if (Editor.tempEdge && Editor.tempEdge.parentNode) {
      Editor.tempEdge.parentNode.removeChild(Editor.tempEdge);
    }
    Editor.tempEdge = null;
    Editor.connect = null;
  }

  function flashError(msg) {
    if (window.TB && window.TB.toast) window.TB.toast(msg, 'error');
  }

  // ---------- 缩放 ----------
  function onWheel(e) {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
    zoomAt(e.clientX, e.clientY, factor);
  }

  function zoomAt(clientX, clientY, factor) {
    const st = window.TB.state;
    const wr = Editor.world.getBoundingClientRect();
    const vr = Editor.viewport.getBoundingClientRect();
    const mx = (clientX - wr.left) / st.zoom;
    const my = (clientY - wr.top) / st.zoom;
    let nz = st.zoom * factor;
    nz = Math.max(0.3, Math.min(2.5, nz));
    st.zoom = nz;
    st.pan.x = clientX - vr.left - mx * nz;
    st.pan.y = clientY - vr.top - my * nz;
    applyTransform();
    if (window.TB.updateZoomLabel) window.TB.updateZoomLabel();
  }

  function zoomBy(factor) {
    const vr = Editor.viewport.getBoundingClientRect();
    zoomAt(vr.left + vr.width / 2, vr.top + vr.height / 2, factor);
  }

  function zoomReset() {
    const st = window.TB.state;
    st.zoom = 1;
    st.pan.x = 0; st.pan.y = 0;
    applyTransform();
    if (window.TB.updateZoomLabel) window.TB.updateZoomLabel();
  }

  function fitView() {
    const { nodes } = window.TB.state;
    if (nodes.length === 0) { zoomReset(); return; }
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    nodes.forEach(n => {
      minX = Math.min(minX, n.x); minY = Math.min(minY, n.y);
      maxX = Math.max(maxX, n.x + 160); maxY = Math.max(maxY, n.y + 80);
    });
    const vr = Editor.viewport.getBoundingClientRect();
    const w = maxX - minX, h = maxY - minY;
    const z = Math.min(2.5, Math.max(0.3, Math.min((vr.width - 40) / w, (vr.height - 40) / h)));
    const st = window.TB.state;
    st.zoom = z;
    st.pan.x = (vr.width - w * z) / 2 - minX * z;
    st.pan.y = (vr.height - h * z) / 2 - minY * z;
    applyTransform();
    if (window.TB.updateZoomLabel) window.TB.updateZoomLabel();
  }

  // ---------- 键盘 ----------
  function setupKeyboard() {
    window.addEventListener('keydown', (e) => {
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable) return;

      // 删除
      if (e.key === 'Delete' || e.key === 'Backspace') {
        const st = window.TB.state;
        if (st.selectedNodeId) { e.preventDefault(); window.TB.removeNode(st.selectedNodeId); }
        else if (st.selectedEdgeId) { e.preventDefault(); window.TB.removeEdge(st.selectedEdgeId); }
        return;
      }
      // 撤销/重做
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault(); window.TB.undo(); return;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y' ||
          ((e.shiftKey) && (e.key === 'z' || e.key === 'Z')))) {
        e.preventDefault(); window.TB.redo(); return;
      }
      // 复制
      if ((e.ctrlKey || e.metaKey) && (e.key === 'd' || e.key === 'D')) {
        e.preventDefault();
        const st = window.TB.state;
        if (st.selectedNodeId) {
          const node = st.nodes.find(n => n.id === st.selectedNodeId);
          if (node) window.TB.addNode(node.type, node.x + 24, node.y + 24, JSON.parse(JSON.stringify(node.params)));
        }
        return;
      }
    });
  }

  // ---------- 初始化 ----------
  function init() {
    Editor.viewport = document.getElementById('tb-viewport');
    Editor.world = document.getElementById('tb-world');
    Editor.nodeLayer = document.getElementById('tb-node-layer');
    Editor.edgeLayer = document.getElementById('tb-edge-layer');
    Editor.edgeGroup = document.getElementById('tb-edge-group');
    applyTransform();
    setupDrop();
    setupCanvas();
    setupKeyboard();
  }

  Editor.init = init;
  Editor.render = render;
  Editor.renderNodes = renderNodes;
  Editor.renderEdges = renderEdges;
  Editor.updateEdges = updateEdges;
  Editor.applyTransform = applyTransform;
  Editor.zoomBy = zoomBy;
  Editor.zoomReset = zoomReset;
  Editor.fitView = fitView;
  Editor.screenToWorld = screenToWorld;

  window.Editor = Editor;
})();

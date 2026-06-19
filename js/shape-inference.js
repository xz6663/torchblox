/* =====================================================================
 * TorchBlox — shape-inference.js
 * 维度推断引擎：Kahn 拓扑排序 + 沿边传播张量形状 + 错误收集。
 * ===================================================================== */
(function () {
  'use strict';

  // 拓扑排序（Kahn 算法），返回 { order, hasCycle }
  function topoSort(nodes, edges) {
    const ids = nodes.map(n => n.id);
    const indeg = {};
    const adj = {}; // nodeId -> [nodeId,...]
    ids.forEach(id => { indeg[id] = 0; adj[id] = []; });
    edges.forEach(e => {
      const f = e.from.nodeId, t = e.to.nodeId;
      if (indeg[t] !== undefined && adj[f] !== undefined) {
        adj[f].push(t);
        indeg[t] += 1;
      }
    });
    const queue = ids.filter(id => indeg[id] === 0);
    const order = [];
    while (queue.length) {
      const cur = queue.shift();
      order.push(cur);
      (adj[cur] || []).forEach(nb => {
        indeg[nb] -= 1;
        if (indeg[nb] === 0) queue.push(nb);
      });
    }
    const hasCycle = order.length < ids.length;
    return { order, hasCycle };
  }

  // 形状比较：两个数组是否逐元素相等（null 视为通配相等）
  function shapesEqual(a, b) {
    if (!a || !b) return false;
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (a[i] == null || b[i] == null) continue;
      if (a[i] !== b[i]) return false;
    }
    return true;
  }

  function shapeRank(s) { return s ? s.length : 0; }

  function fmtShape(s) {
    if (!s || s.length === 0) return '—';
    return '[' + s.map(v => (v == null ? '?' : v)).join(', ') + ']';
  }

  // 推断：返回 { nodeShapes, errors }
  // nodeShapes: { nodeId: { input: [shape,...], output: [shape,...] } }
  // errors: [{ type:'node'|'edge', nodeId?, edgeId?, message }]
  function infer(nodes, edges) {
    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });
    const { order, hasCycle } = topoSort(nodes, edges);

    const nodeShapes = {};
    nodes.forEach(n => { nodeShapes[n.id] = { input: [], output: [] }; });
    const errors = [];

    if (hasCycle) {
      // 找出环中节点（未进入 order 的节点）
      const inOrder = {};
      order.forEach(id => { inOrder[id] = true; });
      nodes.forEach(n => {
        if (!inOrder[n.id]) {
          errors.push({ type: 'node', nodeId: n.id, message: '图中存在环路，无法推断形状' });
        }
      });
      return { nodeShapes, errors };
    }

    // 每个节点的入边（按输入端口索引组织）
    const inEdgesByNode = {};
    nodes.forEach(n => { inEdgesByNode[n.id] = {}; });

    edges.forEach(e => {
      const t = e.to.nodeId;
      if (inEdgesByNode[t]) {
        inEdgesByNode[t][e.to.port] = e;
      }
    });

    // 按拓扑顺序处理
    order.forEach(id => {
      const node = nodeMap[id];
      if (!node) return;
      const def = window.BLOCK_DEFS[node.type];
      if (!def) {
        errors.push({ type: 'node', nodeId: id, message: '未知模块类型：' + node.type });
        return;
      }

      // 收集输入形状
      const inputShapes = [];
      const expectedInputs = def.inputs || 0;
      for (let p = 0; p < expectedInputs; p++) {
        const ein = inEdgesByNode[id][p];
        if (ein) {
          const upstreamOut = nodeShapes[ein.from.nodeId];
          const shp = upstreamOut ? upstreamOut.output : null;
          inputShapes.push(shp ? shp.slice() : null);
        } else {
          inputShapes.push(null);
        }
      }

      // 校验：缺少必要输入
      if (expectedInputs > 0) {
        if (node.type !== 'input') {
          const missing = inputShapes.some(s => s == null);
          if (missing && expectedInputs === 1) {
            errors.push({ type: 'node', nodeId: id, message: '缺少输入连接' });
          } else if (missing) {
            // 合并节点：检查具体缺哪个
            inputShapes.forEach((s, idx) => {
              if (s == null) {
                errors.push({ type: 'node', nodeId: id, message: `输入端口 ${idx} 缺少连接` });
              }
            });
          }
        }
      }

      // 校验：输入秩匹配
      const expectedRank = window.BLOCK_EXPECTED_RANK[node.type];
      if (expectedRank != null && expectedInputs > 0) {
        inputShapes.forEach((s, idx) => {
          if (s && shapeRank(s) !== expectedRank) {
            const ein = inEdgesByNode[id][idx];
            errors.push({
              type: 'edge',
              edgeId: ein ? ein.id : null,
              nodeId: id,
              message: `${def.name} 期望 ${expectedRank}D 输入，实际收到 ${fmtShape(s)}`
            });
          }
        });
      }

      // 合并节点兼容性校验
      if (node.type === 'add' || node.type === 'concat') {
        const a = inputShapes[0], b = inputShapes[1];
        if (a && b) {
          if (node.type === 'add') {
            if (!shapesEqual(a, b)) {
              errors.push({
                type: 'node', nodeId: id,
                message: `Add 要求两输入形状相同（${fmtShape(a)} vs ${fmtShape(b)}）`
              });
            }
          } else {
            // concat
            if (a.length !== b.length) {
              errors.push({
                type: 'node', nodeId: id,
                message: `Concat 要求两输入秩相同（${fmtShape(a)} vs ${fmtShape(b)}）`
              });
            } else {
              let dim = Number(node.params.dim);
              if (dim < 0) dim += a.length;
              for (let d = 0; d < a.length; d++) {
                if (d === dim) continue;
                if (a[d] != null && b[d] != null && a[d] !== b[d]) {
                  errors.push({
                    type: 'node', nodeId: id,
                    message: `Concat 维度 ${d} 不匹配（${a[d]} vs ${b[d]}）`
                  });
                  break;
                }
              }
            }
          }
        }
      }

      // 计算输出形状
      let outShape;
      try {
        outShape = def.shapeFn(inputShapes, node.params || {});
      } catch (err) {
        outShape = [null];
        errors.push({ type: 'node', nodeId: id, message: '形状推断异常：' + err.message });
      }
      nodeShapes[id].input = inputShapes;
      nodeShapes[id].output = (outShape && outShape.length) ? outShape : [];
    });

    return { nodeShapes, errors };
  }

  window.ShapeInference = { topoSort, infer, shapesEqual, fmtShape };
})();

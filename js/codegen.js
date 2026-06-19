/* =====================================================================
 * TorchBlox — codegen.js
 * PyTorch 代码生成：拓扑排序 → 变量命名 → __init__ + forward + 可选训练骨架。
 * forward 采用「主链复用 x、分支点用命名变量」策略，正确处理残差与拼接。
 * ===================================================================== */
(function () {
  'use strict';

  const IND = '    ';          // 4 空格缩进
  const IND2 = IND + IND;

  // 为非 input/output 节点分配变量名：前缀 + 序号
  function assignVarNames(order, nodeMap) {
    const counters = {};
    const names = {};
    order.forEach(id => {
      const node = nodeMap[id];
      if (!node) return;
      if (node.type === 'input' || node.type === 'output') return;
      const prefix = window.BLOCK_VAR_PREFIX[node.type] || 'layer';
      counters[prefix] = (counters[prefix] || 0) + 1;
      names[id] = prefix + counters[prefix];
    });
    return names;
  }

  // 计算每个节点的消费者数量（被多少个节点的输入引用）
  function computeConsumers(nodes, edges) {
    const cnt = {};
    nodes.forEach(n => { cnt[n.id] = 0; });
    edges.forEach(e => {
      if (cnt[e.from.nodeId] !== undefined) cnt[e.from.nodeId] += 1;
    });
    return cnt;
  }

  // 从 input 节点出发 BFS，得到可达节点集合
  function reachableSet(nodes, edges) {
    const adj = {};
    nodes.forEach(n => { adj[n.id] = []; });
    edges.forEach(e => { if (adj[e.from.nodeId]) adj[e.from.nodeId].push(e.to.nodeId); });
    const sources = nodes.filter(n => n.type === 'input').map(n => n.id);
    const seen = {};
    const queue = sources.slice();
    queue.forEach(id => { seen[id] = true; });
    while (queue.length) {
      const cur = queue.shift();
      (adj[cur] || []).forEach(nb => {
        if (!seen[nb]) { seen[nb] = true; queue.push(nb); }
      });
    }
    return seen;
  }

  function generate(nodes, edges, options) {
    options = options || {};
    const modelName = options.modelName || 'MyModel';
    const includeTrainingLoop = !!options.includeTrainingLoop;

    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    const { order, hasCycle } = window.ShapeInference.topoSort(nodes, edges);
    if (hasCycle) {
      return { code: '# 错误：图中存在环路，无法生成代码\n# 请移除环路上的连线后重试。' };
    }

    // 形状推断（用于 codegenInit 自动填充 in_channels 等）
    const inferred = window.ShapeInference.infer(nodes, edges);
    const nodeShapes = inferred.nodeShapes;

    // 收集自定义模块子类定义（在主类之前生成）
    window._pendingCustomClasses = {};
    const varNames = assignVarNames(order, nodeMap);
    const consumers = computeConsumers(nodes, edges);
    const reachable = reachableSet(nodes, edges);

    // 每个节点的入边（按端口索引）
    const inEdgesByNode = {};
    nodes.forEach(n => { inEdgesByNode[n.id] = {}; });
    edges.forEach(e => {
      if (inEdgesByNode[e.to.nodeId]) inEdgesByNode[e.to.nodeId][e.to.port] = e;
    });

    // ---------- __init__ ----------
    const initLines = [];
    initLines.push('super().__init__()');
    order.forEach(id => {
      const node = nodeMap[id];
      if (!node) return;
      if (node.type === 'input' || node.type === 'output') return;
      const def = window.BLOCK_DEFS[node.type];
      if (!def || !def.codegenInit) return;
      const inputShapes = (nodeShapes[id] && nodeShapes[id].input) || [];
      let initStr;
      try {
        initStr = def.codegenInit(node.params || {}, varNames[id], inputShapes);
      } catch (e) {
        initStr = 'nn.Identity()  # 生成失败';
      }
      if (initStr && initStr.trim()) {
        initLines.push(`self.${varNames[id]} = ${initStr}`);
      }
    });

    // ---------- forward ----------
    const fwdLines = [];
    const outVar = {}; // nodeId -> 前向变量名
    let tempCounter = 0;

    // input 节点输出变量：单输入用 x，多输入用 x0, x1, ...
    const inputNodeIds = nodes.filter(n => n.type === 'input').map(n => n.id);
    const multiInput = inputNodeIds.length > 1;
    inputNodeIds.forEach((id, i) => {
      outVar[id] = multiInput ? ('x' + i) : 'x';
    });
    const fwdArgs = multiInput ? inputNodeIds.map((_, i) => 'x' + i).join(', ') : 'x';

    const outputNodes = order.filter(id => nodeMap[id] && nodeMap[id].type === 'output');

    order.forEach(id => {
      const node = nodeMap[id];
      if (!node) return;
      if (node.type === 'input') return;
      if (!reachable[id]) return; // 不可达节点不参与 forward

      const def = window.BLOCK_DEFS[node.type];
      // 收集输入变量
      const inVars = [];
      const expectedInputs = def.inputs || 0;
      let allInputsReady = true;
      for (let p = 0; p < expectedInputs; p++) {
        const ein = inEdgesByNode[id][p];
        if (ein && outVar[ein.from.nodeId] != null) {
          inVars.push(outVar[ein.from.nodeId]);
        } else {
          inVars.push('x');
          allInputsReady = false;
        }
      }

      if (node.type === 'output') {
        // 在最后统一 return，这里跳过
        return;
      }

      // 决定本节点输出变量名
      // 复用上游变量的条件：单输入、非合并、上游仅被本节点消费。
      // 这样重新赋值不会破坏仍被其它节点引用的值；一旦出现分支（上游被多处引用）
      // 即改用命名变量，从而正确支持残差/跳连。
      let myVar;
      const singleInput = expectedInputs === 1;
      const isMerge = expectedInputs > 1;
      const upstreamEdge = inEdgesByNode[id][0];
      const upstreamId = upstreamEdge ? upstreamEdge.from.nodeId : null;
      // 允许复用 'x' 或 'x0'/'x1' 等多输入变量
      const isInputVar = inVars[0] === 'x' || /^x\d+$/.test(inVars[0]);
      const canReuseX = singleInput && !isMerge && isInputVar &&
        upstreamId != null && consumers[upstreamId] === 1;

      if (canReuseX) {
        myVar = inVars[0];  // 复用上游变量名（x 或 x0/x1 等）
      } else {
        myVar = varNames[id] || ('t' + (++tempCounter));
      }

      let expr;
      try {
        expr = def.codegenForward(inVars, node.params || {}, varNames[id]);
      } catch (e) {
        expr = inVars[0];
      }
      fwdLines.push(`${myVar} = ${expr}`);
      outVar[id] = myVar;
    });

    // return 语句
    if (outputNodes.length === 0) {
      // 无显式 output 节点：返回最后一个可达非 input 节点的变量
      let lastVar = 'x';
      order.forEach(id => {
        const node = nodeMap[id];
        if (!node || node.type === 'input') return;
        if (!reachable[id]) return;
        if (outVar[id] != null) lastVar = outVar[id];
      });
      fwdLines.push(`return ${lastVar}`);
    } else if (outputNodes.length === 1) {
      const oid = outputNodes[0];
      const ein = inEdgesByNode[oid][0];
      const v = (ein && outVar[ein.from.nodeId]) || 'x';
      fwdLines.push(`return ${v}`);
    } else {
      const parts = outputNodes.map(oid => {
        const ein = inEdgesByNode[oid][0];
        return (ein && outVar[ein.from.nodeId]) || 'x';
      });
      fwdLines.push(`return ${parts.join(', ')}`);
    }

    // ---------- 生成自定义模块子类 ----------
    const customClassLines = [];
    const customClasses = window._pendingCustomClasses || {};
    Object.keys(customClasses).forEach(cls => {
      const cdef = customClasses[cls];
      const sub = generateCustomClass(cdef, cls);
      customClassLines.push(sub);
      customClassLines.push('');
      customClassLines.push('');
    });

    // ---------- 组装 ----------
    const lines = [];
    lines.push('import torch');
    lines.push('import torch.nn as nn');
    lines.push('import torch.nn.functional as F');
    lines.push('');
    // 自定义模块子类（在主类之前）
    customClassLines.forEach(l => lines.push(l));

    lines.push(`class ${modelName}(nn.Module):`);
    lines.push(`${IND}def __init__(self):`);
    initLines.forEach(l => lines.push(IND2 + l));
    lines.push('');
    lines.push(`${IND}def forward(self, ${fwdArgs}):`);
    if (fwdLines.length === 1) {
      // 只有 return
      lines.push(IND2 + fwdLines[0]);
    } else {
      fwdLines.forEach(l => lines.push(IND2 + l));
    }

    if (includeTrainingLoop) {
      lines.push('');
      lines.push('');
      lines.push("if __name__ == '__main__':");
      lines.push(`${IND}model = ${modelName}()`);
      lines.push(`${IND}criterion = nn.CrossEntropyLoss()`);
      lines.push(`${IND}optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`);
      lines.push('');
      lines.push(`${IND}for epoch in range(5):`);
      lines.push(`${IND2}model.train()`);
      lines.push(`${IND2}for x, y in train_loader:  # 请自行准备 train_loader`);
      lines.push(`${IND2}${IND}optimizer.zero_grad()`);
      lines.push(`${IND2}${IND}out = model(x)`);
      lines.push(`${IND2}${IND}loss = criterion(out, y)`);
      lines.push(`${IND2}${IND}loss.backward()`);
      lines.push(`${IND2}${IND}optimizer.step()`);
      lines.push(`${IND2}print(f'epoch {epoch} done')`);
    }

    return { code: lines.join('\n') };
  }

  // 生成自定义模块的子类代码
  // 结构：class CustomXxx(nn.Module): __init__ + forward
  // 内部子图的 input 节点 → forward 参数；output 节点 → return
  function generateCustomClass(cdef, clsName) {
    const graph = cdef.graph;
    const nodes = graph.nodes;
    const edges = graph.edges;
    const nodeMap = {};
    nodes.forEach(n => { nodeMap[n.id] = n; });

    const { order, hasCycle } = window.ShapeInference.topoSort(nodes, edges);

    // 变量命名（内部）
    const counters = {};
    const varNames = {};
    order.forEach(id => {
      const node = nodeMap[id];
      if (!node) return;
      if (node.type === 'input' || node.type === 'output') return;
      const prefix = window.BLOCK_VAR_PREFIX[node.type] || 'layer';
      counters[prefix] = (counters[prefix] || 0) + 1;
      varNames[id] = prefix + counters[prefix];
    });

    const consumers = computeConsumers(nodes, edges);
    const inEdgesByNode = {};
    nodes.forEach(n => { inEdgesByNode[n.id] = {}; });
    edges.forEach(e => {
      if (inEdgesByNode[e.to.nodeId]) inEdgesByNode[e.to.nodeId][e.to.port] = e;
    });

    // __init__
    const initLines = ['super().__init__()'];
    order.forEach(id => {
      const node = nodeMap[id];
      if (!node) return;
      if (node.type === 'input' || node.type === 'output') return;
      const def = window.BLOCK_DEFS[node.type];
      if (!def || !def.codegenInit) return;
      // 注入自定义参数映射
      const params = JSON.parse(JSON.stringify(node.params || {}));
      if (cdef.paramMapping) {
        cdef.paramMapping.forEach(m => {
          if (m.nodeId === id) {
            // 用默认值（实例化时由外层传入，这里用默认）
            const field = (cdef.paramSchema || []).find(f => f.key === m.blockParam);
            if (field) params[m.nodeParam] = field.default;
          }
        });
      }
      let initStr;
      try {
        initStr = def.codegenInit(params, varNames[id], []);
      } catch (e) {
        initStr = 'nn.Identity()  # 生成失败';
      }
      if (initStr && initStr.trim()) {
        initLines.push(`self.${varNames[id]} = ${initStr}`);
      }
    });

    // forward：input 节点 → 参数名 x0, x1, ...（按 inputBindings 顺序）
    const outVar = {};
    const inputNodes = cdef.inputBindings.map((nid, i) => ({ id: nid, arg: 'x' + (cdef.inputBindings.length > 1 ? i : '') }));
    inputNodes.forEach(it => { outVar[it.id] = it.arg; });

    const outputNodes = order.filter(id => nodeMap[id] && nodeMap[id].type === 'output');
    const fwdLines = [];
    let tempCounter = 0;

    order.forEach(id => {
      const node = nodeMap[id];
      if (!node) return;
      if (node.type === 'input') return;
      const def = window.BLOCK_DEFS[node.type];
      const inVars = [];
      const expectedInputs = def.inputs || 0;
      for (let p = 0; p < expectedInputs; p++) {
        const ein = inEdgesByNode[id][p];
        if (ein && outVar[ein.from.nodeId] != null) inVars.push(outVar[ein.from.nodeId]);
        else inVars.push('x');
      }
      if (node.type === 'output') return;

      let myVar;
      const singleInput = expectedInputs === 1;
      const upstreamEdge = inEdgesByNode[id][0];
      const upstreamId = upstreamEdge ? upstreamEdge.from.nodeId : null;
      const canReuse = singleInput && inVars[0] === outVar[upstreamId] && consumers[upstreamId] === 1;
      if (canReuse) {
        myVar = inVars[0];
      } else {
        myVar = varNames[id] || ('t' + (++tempCounter));
      }

      let expr;
      try {
        expr = def.codegenForward(inVars, node.params || {}, varNames[id]);
      } catch (e) {
        expr = inVars[0];
      }
      fwdLines.push(`${myVar} = ${expr}`);
      outVar[id] = myVar;
    });

    // return
    if (cdef.outputBindings.length === 0) {
      let lastVar = 'x';
      order.forEach(id => {
        if (outVar[id] != null) lastVar = outVar[id];
      });
      fwdLines.push(`return ${lastVar}`);
    } else if (cdef.outputBindings.length === 1) {
      fwdLines.push(`return ${outVar[cdef.outputBindings[0]] || 'x'}`);
    } else {
      const parts = cdef.outputBindings.map(nid => outVar[nid] || 'x');
      fwdLines.push(`return ${parts.join(', ')}`);
    }

    // 组装子类
    const args = cdef.inputBindings.length > 1
      ? cdef.inputBindings.map((_, i) => 'x' + i).join(', ')
      : 'x';
    const lines = [];
    lines.push(`class ${clsName}(nn.Module):`);
    lines.push(`${IND}def __init__(self):`);
    initLines.forEach(l => lines.push(IND2 + l));
    lines.push('');
    lines.push(`${IND}def forward(self, ${args}):`);
    if (fwdLines.length === 1) {
      lines.push(IND2 + fwdLines[0]);
    } else {
      fwdLines.forEach(l => lines.push(IND2 + l));
    }
    return lines.join('\n');
  }

  window.CodeGen = { generate };
})();

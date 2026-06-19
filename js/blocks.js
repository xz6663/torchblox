/* =====================================================================
 * TorchBlox — blocks.js
 * 模块定义与元数据：每个积木包含名称、分类、图标、端口数、默认参数、
 * 参数表单 schema、形状变换函数 shapeFn、代码生成 codegenInit/codegenForward。
 * 形状用数组表示，如 [N,C,H,W]，未知维度用 null。
 * ===================================================================== */
(function () {
  'use strict';

  // 分类顺序与中文标签
  const CATEGORIES = [
    { key: 'input',   label: '输入' },
    { key: 'conv',    label: '卷积' },
    { key: 'pool',    label: '池化' },
    { key: 'norm',    label: '归一化' },
    { key: 'dense',   label: '全连接' },
    { key: 'act',     label: '激活' },
    { key: 'attn',    label: '注意力' },
    { key: 'rnn',     label: '循环' },
    { key: 'reg',     label: '正则化' },
    { key: 'transform', label: '变换' },
    { key: 'merge',   label: '合并' },
    { key: 'output',  label: '输出' }
  ];

  // ---------- 形状计算辅助 ----------
  function convDim(inp, k, s, p) {
    if (inp == null || k == null || s == null || p == null) return null;
    return Math.floor((inp + 2 * p - k) / s) + 1;
  }
  function poolDim(inp, k, s) {
    if (inp == null || k == null || s == null) return null;
    return Math.floor((inp - k) / s) + 1;
  }
  // 解析形如 "[1,3,224,224]" 或 "1,3,224,224" 的文本为数组
  function parseShapeText(txt) {
    if (Array.isArray(txt)) return txt;
    const s = String(txt).replace(/[\[\]\s]/g, '');
    if (!s) return [null];
    return s.split(',').map(v => (v === '' || v === '?' ? null : Number(v)));
  }
  // 末维
  function lastDim(shape) {
    if (!shape || shape.length === 0) return null;
    return shape[shape.length - 1];
  }

  // ---------- 模块定义 ----------
  const DEFS = {
    /* ===================== 输入 ===================== */
    input: {
      type: 'input', name: 'Input', category: 'input', icon: '📥',
      inputs: 0, outputs: 1,
      defaultParams: { shape: '[1, 3, 224, 224]' },
      paramSchema: [
        { key: 'shape', label: '输出形状', type: 'text', default: '[1, 3, 224, 224]' }
      ],
      shapeFn: (ins, p) => parseShapeText(p.shape),
      codegenInit: null,
      codegenForward: null
    },

    /* ===================== 卷积 ===================== */
    conv1d: {
      type: 'conv1d', name: 'Conv1d', category: 'conv', icon: '🎛️',
      inputs: 1, outputs: 1,
      defaultParams: { in_channels: 3, out_channels: 16, kernel_size: 3, stride: 1, padding: 1 },
      paramSchema: [
        { key: 'in_channels', label: '输入通道', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'out_channels', label: '输出通道', type: 'number', min: 1, step: 1, default: 16 },
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 1 },
        { key: 'padding', label: '填充', type: 'number', min: 0, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 3) return [null, Number(p.out_channels), null];
        return [i[0], Number(p.out_channels), convDim(i[2], Number(p.kernel_size), Number(p.stride), Number(p.padding))];
      },
      codegenInit: (p, v, ins) => {
        const inCh = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.in_channels);
        return `nn.Conv1d(in_channels=${inCh}, out_channels=${Number(p.out_channels)}, kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)}, padding=${Number(p.padding)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    conv2d: {
      type: 'conv2d', name: 'Conv2d', category: 'conv', icon: '🔲',
      inputs: 1, outputs: 1,
      defaultParams: { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 },
      paramSchema: [
        { key: 'in_channels', label: '输入通道', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'out_channels', label: '输出通道', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 1 },
        { key: 'padding', label: '填充', type: 'number', min: 0, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 4) return [null, Number(p.out_channels), null, null];
        return [i[0], Number(p.out_channels),
          convDim(i[2], Number(p.kernel_size), Number(p.stride), Number(p.padding)),
          convDim(i[3], Number(p.kernel_size), Number(p.stride), Number(p.padding))];
      },
      codegenInit: (p, v, ins) => {
        const inCh = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.in_channels);
        return `nn.Conv2d(in_channels=${inCh}, out_channels=${Number(p.out_channels)}, kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)}, padding=${Number(p.padding)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    conv3d: {
      type: 'conv3d', name: 'Conv3d', category: 'conv', icon: '🧊',
      inputs: 1, outputs: 1,
      defaultParams: { in_channels: 3, out_channels: 32, kernel_size: 3, stride: 1, padding: 1 },
      paramSchema: [
        { key: 'in_channels', label: '输入通道', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'out_channels', label: '输出通道', type: 'number', min: 1, step: 1, default: 32 },
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 1 },
        { key: 'padding', label: '填充', type: 'number', min: 0, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 5) return [null, Number(p.out_channels), null, null, null];
        const k = Number(p.kernel_size), s = Number(p.stride), pad = Number(p.padding);
        return [i[0], Number(p.out_channels), convDim(i[2], k, s, pad), convDim(i[3], k, s, pad), convDim(i[4], k, s, pad)];
      },
      codegenInit: (p, v, ins) => {
        const inCh = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.in_channels);
        return `nn.Conv3d(in_channels=${inCh}, out_channels=${Number(p.out_channels)}, kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)}, padding=${Number(p.padding)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 池化 ===================== */
    maxpool2d: {
      type: 'maxpool2d', name: 'MaxPool2d', category: 'pool', icon: '⬇️',
      inputs: 1, outputs: 1,
      defaultParams: { kernel_size: 2, stride: 2 },
      paramSchema: [
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 2 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 2 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 4) return i.length ? i.slice() : [null];
        const k = Number(p.kernel_size), s = Number(p.stride);
        return [i[0], i[1], poolDim(i[2], k, s), poolDim(i[3], k, s)];
      },
      codegenInit: (p, v) => `nn.MaxPool2d(kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    avgpool2d: {
      type: 'avgpool2d', name: 'AvgPool2d', category: 'pool', icon: '🔻',
      inputs: 1, outputs: 1,
      defaultParams: { kernel_size: 2, stride: 2 },
      paramSchema: [
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 2 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 2 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 4) return i.length ? i.slice() : [null];
        const k = Number(p.kernel_size), s = Number(p.stride);
        return [i[0], i[1], poolDim(i[2], k, s), poolDim(i[3], k, s)];
      },
      codegenInit: (p, v) => `nn.AvgPool2d(kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    adaptiveavgpool2d: {
      type: 'adaptiveavgpool2d', name: 'AdaptiveAvgPool2d', category: 'pool', icon: '🎯',
      inputs: 1, outputs: 1,
      defaultParams: { output_size: 1 },
      paramSchema: [
        { key: 'output_size', label: '输出尺寸', type: 'number', min: 1, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const o = Number(p.output_size);
        if (i.length < 4) return i.length ? i.slice() : [null];
        return [i[0], i[1], o, o];
      },
      codegenInit: (p, v) => `nn.AdaptiveAvgPool2d(${Number(p.output_size)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 归一化 ===================== */
    batchnorm1d: {
      type: 'batchnorm1d', name: 'BatchNorm1d', category: 'norm', icon: '⚖️',
      inputs: 1, outputs: 1,
      defaultParams: { num_features: 64 },
      paramSchema: [
        { key: 'num_features', label: '特征数', type: 'number', min: 1, step: 1, default: 64 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const nf = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.num_features);
        return `nn.BatchNorm1d(${nf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    batchnorm2d: {
      type: 'batchnorm2d', name: 'BatchNorm2d', category: 'norm', icon: '⚖️',
      inputs: 1, outputs: 1,
      defaultParams: { num_features: 64 },
      paramSchema: [
        { key: 'num_features', label: '特征数', type: 'number', min: 1, step: 1, default: 64 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const nf = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.num_features);
        return `nn.BatchNorm2d(${nf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    layernorm: {
      type: 'layernorm', name: 'LayerNorm', category: 'norm', icon: '📐',
      inputs: 1, outputs: 1,
      defaultParams: { normalized_shape: 64 },
      paramSchema: [
        { key: 'normalized_shape', label: '归一化维度', type: 'number', min: 1, step: 1, default: 64 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const ns = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.normalized_shape);
        return `nn.LayerNorm(${ns})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    groupnorm: {
      type: 'groupnorm', name: 'GroupNorm', category: 'norm', icon: '👥',
      inputs: 1, outputs: 1,
      defaultParams: { num_groups: 8, num_channels: 64 },
      paramSchema: [
        { key: 'num_groups', label: '分组数', type: 'number', min: 1, step: 1, default: 8 },
        { key: 'num_channels', label: '通道数', type: 'number', min: 1, step: 1, default: 64 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const nc = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.num_channels);
        return `nn.GroupNorm(num_groups=${Number(p.num_groups)}, num_channels=${nc})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 全连接 ===================== */
    linear: {
      type: 'linear', name: 'Linear', category: 'dense', icon: '🔗',
      inputs: 1, outputs: 1,
      defaultParams: { in_features: 64, out_features: 10 },
      paramSchema: [
        { key: 'in_features', label: '输入特征', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'out_features', label: '输出特征', type: 'number', min: 1, step: 1, default: 10 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length === 0) return [null, Number(p.out_features)];
        const out = i.slice();
        out[out.length - 1] = Number(p.out_features);
        return out;
      },
      codegenInit: (p, v, ins) => {
        const inF = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.in_features);
        return `nn.Linear(in_features=${inF}, out_features=${Number(p.out_features)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 激活 ===================== */
    relu: {
      type: 'relu', name: 'ReLU', category: 'act', icon: '⚡',
      inputs: 1, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: () => `nn.ReLU()`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    leakyrelu: {
      type: 'leakyrelu', name: 'LeakyReLU', category: 'act', icon: '⚡',
      inputs: 1, outputs: 1,
      defaultParams: { negative_slope: 0.01 },
      paramSchema: [
        { key: 'negative_slope', label: '负斜率', type: 'number', min: 0, step: 0.01, default: 0.01 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.LeakyReLU(negative_slope=${Number(p.negative_slope)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    gelu: {
      type: 'gelu', name: 'GELU', category: 'act', icon: '✨',
      inputs: 1, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: () => `nn.GELU()`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    sigmoid: {
      type: 'sigmoid', name: 'Sigmoid', category: 'act', icon: '🌀',
      inputs: 1, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: () => `nn.Sigmoid()`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    tanh: {
      type: 'tanh', name: 'Tanh', category: 'act', icon: '🌀',
      inputs: 1, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: () => `nn.Tanh()`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    softmax: {
      type: 'softmax', name: 'Softmax', category: 'act', icon: '🔥',
      inputs: 1, outputs: 1,
      defaultParams: { dim: -1 },
      paramSchema: [
        { key: 'dim', label: '计算维度', type: 'number', step: 1, default: -1 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.Softmax(dim=${Number(p.dim)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 注意力 ===================== */
    multiheadattention: {
      type: 'multiheadattention', name: 'MultiheadAttention', category: 'attn', icon: '🧠',
      inputs: 1, outputs: 1,
      defaultParams: { embed_dim: 64, num_heads: 4, batch_first: 'true' },
      paramSchema: [
        { key: 'embed_dim', label: '嵌入维度', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'num_heads', label: '头数', type: 'number', min: 1, step: 1, default: 4 },
        { key: 'batch_first', label: 'batch_first', type: 'select', options: ['true', 'false'], default: 'true' }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const ed = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.embed_dim);
        const bf = String(p.batch_first) === 'true';
        return `nn.MultiheadAttention(embed_dim=${ed}, num_heads=${Number(p.num_heads)}, batch_first=${bf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]}, ${iv[0]}, ${iv[0]}, need_weights=False)[0]`
    },

    /* ===================== 循环 ===================== */
    lstm: {
      type: 'lstm', name: 'LSTM', category: 'rnn', icon: '🔁',
      inputs: 1, outputs: 1,
      defaultParams: { input_size: 64, hidden_size: 128, num_layers: 1, batch_first: 'true' },
      paramSchema: [
        { key: 'input_size', label: '输入维度', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'hidden_size', label: '隐藏维度', type: 'number', min: 1, step: 1, default: 128 },
        { key: 'num_layers', label: '层数', type: 'number', min: 1, step: 1, default: 1 },
        { key: 'batch_first', label: 'batch_first', type: 'select', options: ['true', 'false'], default: 'true' }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const h = Number(p.hidden_size);
        if (i.length < 3) return i.length ? i.slice(0, 2).concat([h]) : [null, null, h];
        return [i[0], i[1], h];
      },
      codegenInit: (p, v, ins) => {
        const isz = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.input_size);
        const bf = String(p.batch_first) === 'true';
        return `nn.LSTM(input_size=${isz}, hidden_size=${Number(p.hidden_size)}, num_layers=${Number(p.num_layers)}, batch_first=${bf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})[0]`
    },
    gru: {
      type: 'gru', name: 'GRU', category: 'rnn', icon: '🔁',
      inputs: 1, outputs: 1,
      defaultParams: { input_size: 64, hidden_size: 128, num_layers: 1, batch_first: 'true' },
      paramSchema: [
        { key: 'input_size', label: '输入维度', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'hidden_size', label: '隐藏维度', type: 'number', min: 1, step: 1, default: 128 },
        { key: 'num_layers', label: '层数', type: 'number', min: 1, step: 1, default: 1 },
        { key: 'batch_first', label: 'batch_first', type: 'select', options: ['true', 'false'], default: 'true' }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const h = Number(p.hidden_size);
        if (i.length < 3) return i.length ? i.slice(0, 2).concat([h]) : [null, null, h];
        return [i[0], i[1], h];
      },
      codegenInit: (p, v, ins) => {
        const isz = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.input_size);
        const bf = String(p.batch_first) === 'true';
        return `nn.GRU(input_size=${isz}, hidden_size=${Number(p.hidden_size)}, num_layers=${Number(p.num_layers)}, batch_first=${bf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})[0]`
    },

    /* ===================== 正则化 ===================== */
    dropout: {
      type: 'dropout', name: 'Dropout', category: 'reg', icon: '💧',
      inputs: 1, outputs: 1,
      defaultParams: { p: 0.5 },
      paramSchema: [
        { key: 'p', label: '丢弃率', type: 'number', min: 0, max: 1, step: 0.05, default: 0.5 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.Dropout(p=${Number(p.p)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 变换 ===================== */
    flatten: {
      type: 'flatten', name: 'Flatten', category: 'transform', icon: '📏',
      inputs: 1, outputs: 1,
      defaultParams: { start_dim: 1, end_dim: -1 },
      paramSchema: [
        { key: 'start_dim', label: '起始维', type: 'number', step: 1, default: 1 },
        { key: 'end_dim', label: '结束维', type: 'number', step: 1, default: -1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length === 0) return [null];
        let start = Number(p.start_dim);
        let end = Number(p.end_dim);
        if (start < 0) start += i.length;
        if (end < 0) end += i.length;
        const out = [];
        for (let d = 0; d < i.length; d++) {
          if (d === start) {
            let prod = 1, anyNull = false;
            for (let e = start; e <= end; e++) {
              if (i[e] == null) anyNull = true; else prod *= i[e];
            }
            out.push(anyNull ? null : prod);
          } else if (d > start && d <= end) {
            // merged
          } else {
            out.push(i[d]);
          }
        }
        return out;
      },
      codegenInit: (p) => `nn.Flatten(start_dim=${Number(p.start_dim)}, end_dim=${Number(p.end_dim)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    reshape: {
      type: 'reshape', name: 'Reshape', category: 'transform', icon: '🧩',
      inputs: 1, outputs: 1,
      defaultParams: { shape: '1, -1' },
      paramSchema: [
        { key: 'shape', label: '目标形状', type: 'text', default: '1, -1' }
      ],
      shapeFn: (ins, p) => {
        const parts = String(p.shape).replace(/[\[\]]/g, '').split(',').map(s => s.trim()).filter(s => s.length);
        return parts.map(s => (s === '-1' || s === '?' ? null : Number(s)));
      },
      codegenInit: null,
      codegenForward: (iv, p) => `${iv[0]}.view(${String(p.shape).replace(/[\[\]]/g, '')})`
    },
    adaptiveavgpool: {
      type: 'adaptiveavgpool', name: 'AdaptiveAvgPool', category: 'transform', icon: '🎯',
      inputs: 1, outputs: 1,
      defaultParams: { output_size: 1 },
      paramSchema: [
        { key: 'output_size', label: '输出尺寸', type: 'number', min: 1, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const o = Number(p.output_size);
        if (i.length < 4) return i.length ? i.slice() : [null];
        return [i[0], i[1], o, o];
      },
      codegenInit: (p) => `nn.AdaptiveAvgPool2d(${Number(p.output_size)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 合并 ===================== */
    concat: {
      type: 'concat', name: 'Concat', category: 'merge', icon: '➕',
      inputs: 2, outputs: 1,
      defaultParams: { dim: 1 },
      paramSchema: [
        { key: 'dim', label: '拼接维度', type: 'number', step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const a = ins[0], b = ins[1];
        if (!a || !b) return a ? a.slice() : (b ? b.slice() : [null]);
        if (a.length !== b.length) return [null];
        let dim = Number(p.dim);
        if (dim < 0) dim += a.length;
        return a.map((v, d) => {
          if (d === dim) return (v == null || b[d] == null) ? null : v + b[d];
          return v;
        });
      },
      codegenInit: null,
      codegenForward: (iv, p) => `torch.cat([${iv[0]}, ${iv[1]}], dim=${Number(p.dim)})`
    },
    add: {
      type: 'add', name: 'Add', category: 'merge', icon: '✚',
      inputs: 2, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : (ins[1] ? ins[1].slice() : [null])),
      codegenInit: null,
      codegenForward: (iv) => `${iv[0]} + ${iv[1]}`
    },

    /* ===================== 转置卷积 ===================== */
    convtranspose1d: {
      type: 'convtranspose1d', name: 'ConvTranspose1d', category: 'conv', icon: '🎛️',
      inputs: 1, outputs: 1,
      defaultParams: { in_channels: 16, out_channels: 3, kernel_size: 3, stride: 1, padding: 0, output_padding: 0 },
      paramSchema: [
        { key: 'in_channels', label: '输入通道', type: 'number', min: 1, step: 1, default: 16 },
        { key: 'out_channels', label: '输出通道', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 1 },
        { key: 'padding', label: '填充', type: 'number', min: 0, step: 1, default: 0 },
        { key: 'output_padding', label: '输出填充', type: 'number', min: 0, step: 1, default: 0 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const k = Number(p.kernel_size), s = Number(p.stride), pad = Number(p.padding), op = Number(p.output_padding);
        if (i.length < 3) return [null, Number(p.out_channels), null];
        return [i[0], Number(p.out_channels), (i[2] - 1) * s - 2 * pad + k + op];
      },
      codegenInit: (p, v, ins) => {
        const inCh = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.in_channels);
        return `nn.ConvTranspose1d(in_channels=${inCh}, out_channels=${Number(p.out_channels)}, kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)}, padding=${Number(p.padding)}, output_padding=${Number(p.output_padding)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    convtranspose2d: {
      type: 'convtranspose2d', name: 'ConvTranspose2d', category: 'conv', icon: '🔲',
      inputs: 1, outputs: 1,
      defaultParams: { in_channels: 64, out_channels: 3, kernel_size: 3, stride: 2, padding: 1, output_padding: 1 },
      paramSchema: [
        { key: 'in_channels', label: '输入通道', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'out_channels', label: '输出通道', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 2 },
        { key: 'padding', label: '填充', type: 'number', min: 0, step: 1, default: 1 },
        { key: 'output_padding', label: '输出填充', type: 'number', min: 0, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const k = Number(p.kernel_size), s = Number(p.stride), pad = Number(p.padding), op = Number(p.output_padding);
        if (i.length < 4) return [null, Number(p.out_channels), null, null];
        return [i[0], Number(p.out_channels),
          (i[2] - 1) * s - 2 * pad + k + op,
          (i[3] - 1) * s - 2 * pad + k + op];
      },
      codegenInit: (p, v, ins) => {
        const inCh = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.in_channels);
        return `nn.ConvTranspose2d(in_channels=${inCh}, out_channels=${Number(p.out_channels)}, kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)}, padding=${Number(p.padding)}, output_padding=${Number(p.output_padding)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    convtranspose3d: {
      type: 'convtranspose3d', name: 'ConvTranspose3d', category: 'conv', icon: '🧊',
      inputs: 1, outputs: 1,
      defaultParams: { in_channels: 32, out_channels: 3, kernel_size: 3, stride: 2, padding: 1, output_padding: 1 },
      paramSchema: [
        { key: 'in_channels', label: '输入通道', type: 'number', min: 1, step: 1, default: 32 },
        { key: 'out_channels', label: '输出通道', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 3 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 2 },
        { key: 'padding', label: '填充', type: 'number', min: 0, step: 1, default: 1 },
        { key: 'output_padding', label: '输出填充', type: 'number', min: 0, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const k = Number(p.kernel_size), s = Number(p.stride), pad = Number(p.padding), op = Number(p.output_padding);
        if (i.length < 5) return [null, Number(p.out_channels), null, null, null];
        return [i[0], Number(p.out_channels),
          (i[2] - 1) * s - 2 * pad + k + op,
          (i[3] - 1) * s - 2 * pad + k + op,
          (i[4] - 1) * s - 2 * pad + k + op];
      },
      codegenInit: (p, v, ins) => {
        const inCh = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.in_channels);
        return `nn.ConvTranspose3d(in_channels=${inCh}, out_channels=${Number(p.out_channels)}, kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)}, padding=${Number(p.padding)}, output_padding=${Number(p.output_padding)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 池化补充 ===================== */
    maxpool1d: {
      type: 'maxpool1d', name: 'MaxPool1d', category: 'pool', icon: '⬇️',
      inputs: 1, outputs: 1,
      defaultParams: { kernel_size: 2, stride: 2 },
      paramSchema: [
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 2 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 2 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 3) return i.length ? i.slice() : [null];
        return [i[0], i[1], poolDim(i[2], Number(p.kernel_size), Number(p.stride))];
      },
      codegenInit: (p) => `nn.MaxPool1d(kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    maxpool3d: {
      type: 'maxpool3d', name: 'MaxPool3d', category: 'pool', icon: '⬇️',
      inputs: 1, outputs: 1,
      defaultParams: { kernel_size: 2, stride: 2 },
      paramSchema: [
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 2 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 2 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 5) return i.length ? i.slice() : [null];
        const k = Number(p.kernel_size), s = Number(p.stride);
        return [i[0], i[1], poolDim(i[2], k, s), poolDim(i[3], k, s), poolDim(i[4], k, s)];
      },
      codegenInit: (p) => `nn.MaxPool3d(kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    avgpool1d: {
      type: 'avgpool1d', name: 'AvgPool1d', category: 'pool', icon: '🔻',
      inputs: 1, outputs: 1,
      defaultParams: { kernel_size: 2, stride: 2 },
      paramSchema: [
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 2 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 2 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 3) return i.length ? i.slice() : [null];
        return [i[0], i[1], poolDim(i[2], Number(p.kernel_size), Number(p.stride))];
      },
      codegenInit: (p) => `nn.AvgPool1d(kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    avgpool3d: {
      type: 'avgpool3d', name: 'AvgPool3d', category: 'pool', icon: '🔻',
      inputs: 1, outputs: 1,
      defaultParams: { kernel_size: 2, stride: 2 },
      paramSchema: [
        { key: 'kernel_size', label: '核大小', type: 'number', min: 1, step: 1, default: 2 },
        { key: 'stride', label: '步长', type: 'number', min: 1, step: 1, default: 2 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 5) return i.length ? i.slice() : [null];
        const k = Number(p.kernel_size), s = Number(p.stride);
        return [i[0], i[1], poolDim(i[2], k, s), poolDim(i[3], k, s), poolDim(i[4], k, s)];
      },
      codegenInit: (p) => `nn.AvgPool3d(kernel_size=${Number(p.kernel_size)}, stride=${Number(p.stride)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    adaptivemaxpool2d: {
      type: 'adaptivemaxpool2d', name: 'AdaptiveMaxPool2d', category: 'pool', icon: '🎯',
      inputs: 1, outputs: 1,
      defaultParams: { output_size: 1 },
      paramSchema: [
        { key: 'output_size', label: '输出尺寸', type: 'number', min: 1, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const o = Number(p.output_size);
        if (i.length < 4) return i.length ? i.slice() : [null];
        return [i[0], i[1], o, o];
      },
      codegenInit: (p) => `nn.AdaptiveMaxPool2d(${Number(p.output_size)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    adaptiveavgpool1d: {
      type: 'adaptiveavgpool1d', name: 'AdaptiveAvgPool1d', category: 'pool', icon: '🎯',
      inputs: 1, outputs: 1,
      defaultParams: { output_size: 1 },
      paramSchema: [
        { key: 'output_size', label: '输出尺寸', type: 'number', min: 1, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length < 3) return i.length ? i.slice() : [null];
        return [i[0], i[1], Number(p.output_size)];
      },
      codegenInit: (p) => `nn.AdaptiveAvgPool1d(${Number(p.output_size)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    adaptiveavgpool3d: {
      type: 'adaptiveavgpool3d', name: 'AdaptiveAvgPool3d', category: 'pool', icon: '🎯',
      inputs: 1, outputs: 1,
      defaultParams: { output_size: 1 },
      paramSchema: [
        { key: 'output_size', label: '输出尺寸', type: 'number', min: 1, step: 1, default: 1 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const o = Number(p.output_size);
        if (i.length < 5) return i.length ? i.slice() : [null];
        return [i[0], i[1], o, o, o];
      },
      codegenInit: (p) => `nn.AdaptiveAvgPool3d(${Number(p.output_size)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 归一化补充 ===================== */
    batchnorm3d: {
      type: 'batchnorm3d', name: 'BatchNorm3d', category: 'norm', icon: '⚖️',
      inputs: 1, outputs: 1,
      defaultParams: { num_features: 32 },
      paramSchema: [
        { key: 'num_features', label: '特征数', type: 'number', min: 1, step: 1, default: 32 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const nf = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.num_features);
        return `nn.BatchNorm3d(${nf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    instancenorm1d: {
      type: 'instancenorm1d', name: 'InstanceNorm1d', category: 'norm', icon: '⚖️',
      inputs: 1, outputs: 1,
      defaultParams: { num_features: 64 },
      paramSchema: [
        { key: 'num_features', label: '特征数', type: 'number', min: 1, step: 1, default: 64 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const nf = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.num_features);
        return `nn.InstanceNorm1d(${nf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    instancenorm2d: {
      type: 'instancenorm2d', name: 'InstanceNorm2d', category: 'norm', icon: '⚖️',
      inputs: 1, outputs: 1,
      defaultParams: { num_features: 64 },
      paramSchema: [
        { key: 'num_features', label: '特征数', type: 'number', min: 1, step: 1, default: 64 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const nf = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.num_features);
        return `nn.InstanceNorm2d(${nf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    instancenorm3d: {
      type: 'instancenorm3d', name: 'InstanceNorm3d', category: 'norm', icon: '⚖️',
      inputs: 1, outputs: 1,
      defaultParams: { num_features: 32 },
      paramSchema: [
        { key: 'num_features', label: '特征数', type: 'number', min: 1, step: 1, default: 32 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const nf = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.num_features);
        return `nn.InstanceNorm3d(${nf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    syncbatchnorm: {
      type: 'syncbatchnorm', name: 'SyncBatchNorm', category: 'norm', icon: '⚖️',
      inputs: 1, outputs: 1,
      defaultParams: { num_features: 64 },
      paramSchema: [
        { key: 'num_features', label: '特征数', type: 'number', min: 1, step: 1, default: 64 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const nf = (ins[0] && ins[0][1] != null) ? ins[0][1] : Number(p.num_features);
        return `nn.SyncBatchNorm(${nf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 激活补充 ===================== */
    elu: {
      type: 'elu', name: 'ELU', category: 'act', icon: '⚡',
      inputs: 1, outputs: 1,
      defaultParams: { alpha: 1.0 },
      paramSchema: [
        { key: 'alpha', label: 'alpha', type: 'number', min: 0, step: 0.1, default: 1.0 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.ELU(alpha=${Number(p.alpha)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    prelu: {
      type: 'prelu', name: 'PReLU', category: 'act', icon: '⚡',
      inputs: 1, outputs: 1,
      defaultParams: { num_parameters: 1, init: 0.25 },
      paramSchema: [
        { key: 'num_parameters', label: '参数数', type: 'number', min: 1, step: 1, default: 1 },
        { key: 'init', label: '初始值', type: 'number', step: 0.05, default: 0.25 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.PReLU(num_parameters=${Number(p.num_parameters)}, init=${Number(p.init)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    relu6: {
      type: 'relu6', name: 'ReLU6', category: 'act', icon: '⚡',
      inputs: 1, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: () => `nn.ReLU6()`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    selu: {
      type: 'selu', name: 'SELU', category: 'act', icon: '✨',
      inputs: 1, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: () => `nn.SELU()`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    hardswish: {
      type: 'hardswish', name: 'Hardswish', category: 'act', icon: '⚡',
      inputs: 1, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: () => `nn.Hardswish()`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    hardsigmoid: {
      type: 'hardsigmoid', name: 'Hardsigmoid', category: 'act', icon: '🌀',
      inputs: 1, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: () => `nn.Hardsigmoid()`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    hardtanh: {
      type: 'hardtanh', name: 'Hardtanh', category: 'act', icon: '⚡',
      inputs: 1, outputs: 1,
      defaultParams: { min_val: -1.0, max_val: 1.0 },
      paramSchema: [
        { key: 'min_val', label: '最小值', type: 'number', step: 0.1, default: -1.0 },
        { key: 'max_val', label: '最大值', type: 'number', step: 0.1, default: 1.0 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.Hardtanh(min_val=${Number(p.min_val)}, max_val=${Number(p.max_val)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    logsoftmax: {
      type: 'logsoftmax', name: 'LogSoftmax', category: 'act', icon: '🔥',
      inputs: 1, outputs: 1,
      defaultParams: { dim: -1 },
      paramSchema: [
        { key: 'dim', label: '计算维度', type: 'number', step: 1, default: -1 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.LogSoftmax(dim=${Number(p.dim)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    softmin: {
      type: 'softmin', name: 'Softmin', category: 'act', icon: '🔥',
      inputs: 1, outputs: 1,
      defaultParams: { dim: -1 },
      paramSchema: [
        { key: 'dim', label: '计算维度', type: 'number', step: 1, default: -1 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.Softmin(dim=${Number(p.dim)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 循环补充 ===================== */
    rnn: {
      type: 'rnn', name: 'RNN', category: 'rnn', icon: '🔁',
      inputs: 1, outputs: 1,
      defaultParams: { input_size: 64, hidden_size: 128, num_layers: 1, batch_first: 'true' },
      paramSchema: [
        { key: 'input_size', label: '输入维度', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'hidden_size', label: '隐藏维度', type: 'number', min: 1, step: 1, default: 128 },
        { key: 'num_layers', label: '层数', type: 'number', min: 1, step: 1, default: 1 },
        { key: 'batch_first', label: 'batch_first', type: 'select', options: ['true', 'false'], default: 'true' }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const h = Number(p.hidden_size);
        if (i.length < 3) return i.length ? i.slice(0, 2).concat([h]) : [null, null, h];
        return [i[0], i[1], h];
      },
      codegenInit: (p, v, ins) => {
        const isz = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.input_size);
        const bf = String(p.batch_first) === 'true';
        return `nn.RNN(input_size=${isz}, hidden_size=${Number(p.hidden_size)}, num_layers=${Number(p.num_layers)}, batch_first=${bf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})[0]`
    },
    rnncell: {
      type: 'rnncell', name: 'RNNCell', category: 'rnn', icon: '🔁',
      inputs: 1, outputs: 1,
      defaultParams: { input_size: 64, hidden_size: 128 },
      paramSchema: [
        { key: 'input_size', label: '输入维度', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'hidden_size', label: '隐藏维度', type: 'number', min: 1, step: 1, default: 128 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const h = Number(p.hidden_size);
        if (i.length === 0) return [null, h];
        return i.slice(0, -1).concat([h]);
      },
      codegenInit: (p, v, ins) => {
        const isz = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.input_size);
        return `nn.RNNCell(input_size=${isz}, hidden_size=${Number(p.hidden_size)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    lstmcell: {
      type: 'lstmcell', name: 'LSTMCell', category: 'rnn', icon: '🔁',
      inputs: 1, outputs: 1,
      defaultParams: { input_size: 64, hidden_size: 128 },
      paramSchema: [
        { key: 'input_size', label: '输入维度', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'hidden_size', label: '隐藏维度', type: 'number', min: 1, step: 1, default: 128 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const h = Number(p.hidden_size);
        if (i.length === 0) return [null, h];
        return i.slice(0, -1).concat([h]);
      },
      codegenInit: (p, v, ins) => {
        const isz = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.input_size);
        return `nn.LSTMCell(input_size=${isz}, hidden_size=${Number(p.hidden_size)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    grucell: {
      type: 'grucell', name: 'GRUCell', category: 'rnn', icon: '🔁',
      inputs: 1, outputs: 1,
      defaultParams: { input_size: 64, hidden_size: 128 },
      paramSchema: [
        { key: 'input_size', label: '输入维度', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'hidden_size', label: '隐藏维度', type: 'number', min: 1, step: 1, default: 128 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const h = Number(p.hidden_size);
        if (i.length === 0) return [null, h];
        return i.slice(0, -1).concat([h]);
      },
      codegenInit: (p, v, ins) => {
        const isz = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.input_size);
        return `nn.GRUCell(input_size=${isz}, hidden_size=${Number(p.hidden_size)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== Transformer ===================== */
    transformerencoderlayer: {
      type: 'transformerencoderlayer', name: 'TransformerEncoderLayer', category: 'attn', icon: '🧠',
      inputs: 1, outputs: 1,
      defaultParams: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1, batch_first: 'true' },
      paramSchema: [
        { key: 'd_model', label: '模型维度', type: 'number', min: 1, step: 1, default: 512 },
        { key: 'nhead', label: '头数', type: 'number', min: 1, step: 1, default: 8 },
        { key: 'dim_feedforward', label: 'FFN 维度', type: 'number', min: 1, step: 1, default: 2048 },
        { key: 'dropout', label: '丢弃率', type: 'number', min: 0, max: 1, step: 0.05, default: 0.1 },
        { key: 'batch_first', label: 'batch_first', type: 'select', options: ['true', 'false'], default: 'true' }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const dm = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.d_model);
        const bf = String(p.batch_first) === 'true';
        return `nn.TransformerEncoderLayer(d_model=${dm}, nhead=${Number(p.nhead)}, dim_feedforward=${Number(p.dim_feedforward)}, dropout=${Number(p.dropout)}, batch_first=${bf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    transformerdecoderlayer: {
      type: 'transformerdecoderlayer', name: 'TransformerDecoderLayer', category: 'attn', icon: '🧠',
      inputs: 2, outputs: 1,
      defaultParams: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1, batch_first: 'true' },
      paramSchema: [
        { key: 'd_model', label: '模型维度', type: 'number', min: 1, step: 1, default: 512 },
        { key: 'nhead', label: '头数', type: 'number', min: 1, step: 1, default: 8 },
        { key: 'dim_feedforward', label: 'FFN 维度', type: 'number', min: 1, step: 1, default: 2048 },
        { key: 'dropout', label: '丢弃率', type: 'number', min: 0, max: 1, step: 0.05, default: 0.1 },
        { key: 'batch_first', label: 'batch_first', type: 'select', options: ['true', 'false'], default: 'true' }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const dm = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.d_model);
        const bf = String(p.batch_first) === 'true';
        return `nn.TransformerDecoderLayer(d_model=${dm}, nhead=${Number(p.nhead)}, dim_feedforward=${Number(p.dim_feedforward)}, dropout=${Number(p.dropout)}, batch_first=${bf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]}, ${iv[1]})`
    },
    transformerencoder: {
      type: 'transformerencoder', name: 'TransformerEncoder', category: 'attn', icon: '🧠',
      inputs: 1, outputs: 1,
      defaultParams: { d_model: 512, nhead: 8, num_layers: 6, dim_feedforward: 2048, dropout: 0.1, batch_first: 'true' },
      paramSchema: [
        { key: 'd_model', label: '模型维度', type: 'number', min: 1, step: 1, default: 512 },
        { key: 'nhead', label: '头数', type: 'number', min: 1, step: 1, default: 8 },
        { key: 'num_layers', label: '层数', type: 'number', min: 1, step: 1, default: 6 },
        { key: 'dim_feedforward', label: 'FFN 维度', type: 'number', min: 1, step: 1, default: 2048 },
        { key: 'dropout', label: '丢弃率', type: 'number', min: 0, max: 1, step: 0.05, default: 0.1 },
        { key: 'batch_first', label: 'batch_first', type: 'select', options: ['true', 'false'], default: 'true' }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const dm = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.d_model);
        const bf = String(p.batch_first) === 'true';
        const layer = `nn.TransformerEncoderLayer(d_model=${dm}, nhead=${Number(p.nhead)}, dim_feedforward=${Number(p.dim_feedforward)}, dropout=${Number(p.dropout)}, batch_first=${bf})`;
        return `nn.TransformerEncoder(${layer}, num_layers=${Number(p.num_layers)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    transformerdecoder: {
      type: 'transformerdecoder', name: 'TransformerDecoder', category: 'attn', icon: '🧠',
      inputs: 2, outputs: 1,
      defaultParams: { d_model: 512, nhead: 8, num_layers: 6, dim_feedforward: 2048, dropout: 0.1, batch_first: 'true' },
      paramSchema: [
        { key: 'd_model', label: '模型维度', type: 'number', min: 1, step: 1, default: 512 },
        { key: 'nhead', label: '头数', type: 'number', min: 1, step: 1, default: 8 },
        { key: 'num_layers', label: '层数', type: 'number', min: 1, step: 1, default: 6 },
        { key: 'dim_feedforward', label: 'FFN 维度', type: 'number', min: 1, step: 1, default: 2048 },
        { key: 'dropout', label: '丢弃率', type: 'number', min: 0, max: 1, step: 0.05, default: 0.1 },
        { key: 'batch_first', label: 'batch_first', type: 'select', options: ['true', 'false'], default: 'true' }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const dm = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.d_model);
        const bf = String(p.batch_first) === 'true';
        const layer = `nn.TransformerDecoderLayer(d_model=${dm}, nhead=${Number(p.nhead)}, dim_feedforward=${Number(p.dim_feedforward)}, dropout=${Number(p.dropout)}, batch_first=${bf})`;
        return `nn.TransformerDecoder(${layer}, num_layers=${Number(p.num_layers)})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]}, ${iv[1]})`
    },
    transformer: {
      type: 'transformer', name: 'Transformer', category: 'attn', icon: '🧠',
      inputs: 2, outputs: 1,
      defaultParams: { d_model: 512, nhead: 8, num_encoder_layers: 6, num_decoder_layers: 6, dim_feedforward: 2048, dropout: 0.1, batch_first: 'true' },
      paramSchema: [
        { key: 'd_model', label: '模型维度', type: 'number', min: 1, step: 1, default: 512 },
        { key: 'nhead', label: '头数', type: 'number', min: 1, step: 1, default: 8 },
        { key: 'num_encoder_layers', label: '编码层数', type: 'number', min: 1, step: 1, default: 6 },
        { key: 'num_decoder_layers', label: '解码层数', type: 'number', min: 1, step: 1, default: 6 },
        { key: 'dim_feedforward', label: 'FFN 维度', type: 'number', min: 1, step: 1, default: 2048 },
        { key: 'dropout', label: '丢弃率', type: 'number', min: 0, max: 1, step: 0.05, default: 0.1 },
        { key: 'batch_first', label: 'batch_first', type: 'select', options: ['true', 'false'], default: 'true' }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p, v, ins) => {
        const dm = (ins[0] && lastDim(ins[0]) != null) ? lastDim(ins[0]) : Number(p.d_model);
        const bf = String(p.batch_first) === 'true';
        return `nn.Transformer(d_model=${dm}, nhead=${Number(p.nhead)}, num_encoder_layers=${Number(p.num_encoder_layers)}, num_decoder_layers=${Number(p.num_decoder_layers)}, dim_feedforward=${Number(p.dim_feedforward)}, dropout=${Number(p.dropout)}, batch_first=${bf})`;
      },
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]}, ${iv[1]})`
    },

    /* ===================== 稀疏（嵌入） ===================== */
    embedding: {
      type: 'embedding', name: 'Embedding', category: 'dense', icon: '📚',
      inputs: 1, outputs: 1,
      defaultParams: { num_embeddings: 1000, embedding_dim: 128 },
      paramSchema: [
        { key: 'num_embeddings', label: '词表大小', type: 'number', min: 1, step: 1, default: 1000 },
        { key: 'embedding_dim', label: '嵌入维度', type: 'number', min: 1, step: 1, default: 128 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length === 0) return [null, Number(p.embedding_dim)];
        return i.concat([Number(p.embedding_dim)]);
      },
      codegenInit: (p) => `nn.Embedding(num_embeddings=${Number(p.num_embeddings)}, embedding_dim=${Number(p.embedding_dim)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]}.long())`
    },
    embeddingbag: {
      type: 'embeddingbag', name: 'EmbeddingBag', category: 'dense', icon: '📚',
      inputs: 1, outputs: 1,
      defaultParams: { num_embeddings: 1000, embedding_dim: 128, mode: 'mean' },
      paramSchema: [
        { key: 'num_embeddings', label: '词表大小', type: 'number', min: 1, step: 1, default: 1000 },
        { key: 'embedding_dim', label: '嵌入维度', type: 'number', min: 1, step: 1, default: 128 },
        { key: 'mode', label: '聚合模式', type: 'select', options: ['sum', 'mean', 'max'], default: 'mean' }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length === 0) return [null, Number(p.embedding_dim)];
        return i.slice(0, -1).concat([Number(p.embedding_dim)]);
      },
      codegenInit: (p) => `nn.EmbeddingBag(num_embeddings=${Number(p.num_embeddings)}, embedding_dim=${Number(p.embedding_dim)}, mode="${String(p.mode)}")`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 视觉变换 ===================== */
    upsample: {
      type: 'upsample', name: 'Upsample', category: 'transform', icon: '⬆️',
      inputs: 1, outputs: 1,
      defaultParams: { scale_factor: 2, mode: 'nearest' },
      paramSchema: [
        { key: 'scale_factor', label: '缩放因子', type: 'number', min: 0.1, step: 0.5, default: 2 },
        { key: 'mode', label: '插值模式', type: 'select', options: ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'], default: 'nearest' }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const sf = Number(p.scale_factor);
        if (i.length < 3) return i.length ? i.slice() : [null];
        const out = i.slice(0, 2);
        for (let d = 2; d < i.length; d++) out.push(i[d] == null ? null : Math.floor(i[d] * sf));
        return out;
      },
      codegenInit: (p) => `nn.Upsample(scale_factor=${Number(p.scale_factor)}, mode="${String(p.mode)}")`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    pixelshuffle: {
      type: 'pixelshuffle', name: 'PixelShuffle', category: 'transform', icon: '🧩',
      inputs: 1, outputs: 1,
      defaultParams: { upscale_factor: 2 },
      paramSchema: [
        { key: 'upscale_factor', label: '上采样因子', type: 'number', min: 1, step: 1, default: 2 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const r = Number(p.upscale_factor);
        if (i.length < 4) return i.length ? i.slice() : [null];
        return [i[0], i[1] == null ? null : i[1] / (r * r), i[2] == null ? null : i[2] * r, i[3] == null ? null : i[3] * r];
      },
      codegenInit: (p) => `nn.PixelShuffle(upscale_factor=${Number(p.upscale_factor)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    pixelunshuffle: {
      type: 'pixelunshuffle', name: 'PixelUnshuffle', category: 'transform', icon: '🧩',
      inputs: 1, outputs: 1,
      defaultParams: { downscale_factor: 2 },
      paramSchema: [
        { key: 'downscale_factor', label: '下采样因子', type: 'number', min: 1, step: 1, default: 2 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        const r = Number(p.downscale_factor);
        if (i.length < 4) return i.length ? i.slice() : [null];
        return [i[0], i[1] == null ? null : i[1] * (r * r), i[2] == null ? null : i[2] / r, i[3] == null ? null : i[3] / r];
      },
      codegenInit: (p) => `nn.PixelUnshuffle(downscale_factor=${Number(p.downscale_factor)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== Dropout 补充 ===================== */
    dropout2d: {
      type: 'dropout2d', name: 'Dropout2d', category: 'reg', icon: '💧',
      inputs: 1, outputs: 1,
      defaultParams: { p: 0.5 },
      paramSchema: [
        { key: 'p', label: '丢弃率', type: 'number', min: 0, max: 1, step: 0.05, default: 0.5 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.Dropout2d(p=${Number(p.p)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    alphadropout: {
      type: 'alphadropout', name: 'AlphaDropout', category: 'reg', icon: '💧',
      inputs: 1, outputs: 1,
      defaultParams: { p: 0.5 },
      paramSchema: [
        { key: 'p', label: '丢弃率', type: 'number', min: 0, max: 1, step: 0.05, default: 0.5 }
      ],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: (p) => `nn.AlphaDropout(p=${Number(p.p)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 全连接补充 ===================== */
    bilinear: {
      type: 'bilinear', name: 'Bilinear', category: 'dense', icon: '🔗',
      inputs: 2, outputs: 1,
      defaultParams: { in1_features: 64, in2_features: 64, out_features: 10 },
      paramSchema: [
        { key: 'in1_features', label: '输入1特征', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'in2_features', label: '输入2特征', type: 'number', min: 1, step: 1, default: 64 },
        { key: 'out_features', label: '输出特征', type: 'number', min: 1, step: 1, default: 10 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length === 0) return [null, Number(p.out_features)];
        const out = i.slice(); out[out.length - 1] = Number(p.out_features); return out;
      },
      codegenInit: (p) => `nn.Bilinear(in1_features=${Number(p.in1_features)}, in2_features=${Number(p.in2_features)}, out_features=${Number(p.out_features)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]}, ${iv[1]})`
    },
    identity: {
      type: 'identity', name: 'Identity', category: 'transform', icon: '⏭️',
      inputs: 1, outputs: 1, defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: () => `nn.Identity()`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },
    lazylinear: {
      type: 'lazylinear', name: 'LazyLinear', category: 'dense', icon: '🔗',
      inputs: 1, outputs: 1,
      defaultParams: { out_features: 10 },
      paramSchema: [
        { key: 'out_features', label: '输出特征', type: 'number', min: 1, step: 1, default: 10 }
      ],
      shapeFn: (ins, p) => {
        const i = ins[0] || [];
        if (i.length === 0) return [null, Number(p.out_features)];
        const out = i.slice(); out[out.length - 1] = Number(p.out_features); return out;
      },
      codegenInit: (p) => `nn.LazyLinear(out_features=${Number(p.out_features)})`,
      codegenForward: (iv, p, v) => `self.${v}(${iv[0]})`
    },

    /* ===================== 输出 ===================== */
    output: {
      type: 'output', name: 'Output', category: 'output', icon: '📤',
      inputs: 1, outputs: 0,
      defaultParams: {}, paramSchema: [],
      shapeFn: (ins) => (ins[0] ? ins[0].slice() : [null]),
      codegenInit: null,
      codegenForward: null
    }
  };

  // 变量名前缀映射（用于代码生成 self.xxx 命名）
  const VAR_PREFIX = {
    conv1d: 'conv', conv2d: 'conv', conv3d: 'conv',
    convtranspose1d: 'deconv', convtranspose2d: 'deconv', convtranspose3d: 'deconv',
    maxpool1d: 'pool', maxpool2d: 'pool', maxpool3d: 'pool',
    avgpool1d: 'pool', avgpool2d: 'pool', avgpool3d: 'pool',
    adaptiveavgpool1d: 'pool', adaptiveavgpool2d: 'pool', adaptiveavgpool3d: 'pool', adaptiveavgpool: 'pool',
    adaptivemaxpool2d: 'pool',
    batchnorm1d: 'bn', batchnorm2d: 'bn', batchnorm3d: 'bn', syncbatchnorm: 'sbn',
    instancenorm1d: 'in', instancenorm2d: 'in', instancenorm3d: 'in',
    layernorm: 'ln', groupnorm: 'gn',
    linear: 'fc', lazylinear: 'fc', bilinear: 'bi',
    relu: 'relu', leakyrelu: 'relu', relu6: 'relu', elu: 'elu', selu: 'selu',
    prelu: 'prelu', gelu: 'gelu', sigmoid: 'sig', hardsigmoid: 'sig', tanh: 'tanh',
    hardswish: 'hw', hardtanh: 'ht', softmax: 'sm', logsoftmax: 'sm', softmin: 'sm',
    multiheadattention: 'attn',
    transformerencoderlayer: 'enc', transformerdecoderlayer: 'dec',
    transformerencoder: 'enc', transformerdecoder: 'dec', transformer: 'tfm',
    rnn: 'rnn', rnncell: 'rnncell', lstm: 'lstm', lstmcell: 'lstmcell', gru: 'gru', grucell: 'grucell',
    dropout: 'drop', dropout2d: 'drop', alphadropout: 'drop',
    embedding: 'emb', embeddingbag: 'emb',
    flatten: 'flat', reshape: 'view', upsample: 'up', pixelshuffle: 'ps', pixelunshuffle: 'pus', identity: 'id',
    concat: 'cat', add: 'add'
  };

  // 预期输入秩（用于形状校验）；null 表示任意
  const EXPECTED_RANK = {
    conv1d: 3, conv2d: 4, conv3d: 5,
    convtranspose1d: 3, convtranspose2d: 4, convtranspose3d: 5,
    maxpool1d: 3, maxpool2d: 4, maxpool3d: 5,
    avgpool1d: 3, avgpool2d: 4, avgpool3d: 5,
    adaptiveavgpool1d: 3, adaptiveavgpool2d: 4, adaptiveavgpool3d: 5, adaptiveavgpool: 4,
    adaptivemaxpool2d: 4,
    batchnorm1d: null, batchnorm2d: 4, batchnorm3d: 5, syncbatchnorm: 4,
    instancenorm1d: 3, instancenorm2d: 4, instancenorm3d: 5,
    layernorm: null, groupnorm: null,
    linear: null, lazylinear: null, bilinear: null,
    relu: null, leakyrelu: null, relu6: null, elu: null, selu: null, prelu: null,
    gelu: null, sigmoid: null, hardsigmoid: null, tanh: null, hardtanh: null,
    hardswish: null, softmax: null, logsoftmax: null, softmin: null,
    multiheadattention: 3,
    transformerencoderlayer: 3, transformerdecoderlayer: 3,
    transformerencoder: 3, transformerdecoder: 3, transformer: 3,
    rnn: 3, rnncell: null, lstm: 3, lstmcell: null, gru: 3, grucell: null,
    dropout: null, dropout2d: 4, alphadropout: null,
    embedding: null, embeddingbag: null,
    flatten: null, reshape: null, upsample: null, pixelshuffle: 4, pixelunshuffle: 4, identity: null,
    concat: null, add: null,
    output: null, input: null
  };

  // ---------- 模块介绍描述 ----------
  const DESCRIPTIONS = {
    input: {
      summary: '模型输入入口，定义输入张量的形状。',
      role: '作为计算图的起点，指定数据的维度（如 [N, C, H, W]），后续模块据此推断形状。',
      uses: '每个模型必须有一个 Input 节点；常用于指定图像、序列或特征向量的输入维度。'
    },
    conv1d: {
      summary: '一维卷积层，沿序列方向滑动提取局部特征。',
      role: '通过可学习的卷积核在一维输入上做局部加权求和，输出通道数由 out_channels 决定。',
      uses: '文本分类、时间序列预测、一维信号处理（如音频、传感器数据）。'
    },
    conv2d: {
      summary: '二维卷积层，深度学习视觉任务的核心模块。',
      role: '在 H×W 特征图上滑动卷积核，提取空间局部特征（边缘、纹理、形状等），同时改变通道数。',
      uses: '图像分类、目标检测、语义分割等几乎所有 CNN 架构（ResNet、VGG、LeNet 等）。'
    },
    conv3d: {
      summary: '三维卷积层，在空间+深度三个维度上提取特征。',
      role: '卷积核在 (D, H, W) 三个维度上滑动，捕获立体局部特征，参数量较大。',
      uses: '医学影像（CT/MRI）、视频动作识别、3D 点云体素化后的特征提取。'
    },
    maxpool2d: {
      summary: '最大池化层，下采样并保留最显著特征。',
      role: '在每个池化窗口内取最大值，缩小特征图尺寸、增加平移不变性、减少计算量。',
      uses: 'CNN 中常用的下采样手段，通常接在卷积+激活之后，逐步降低空间分辨率。'
    },
    avgpool2d: {
      summary: '平均池化层，下采样并平滑特征。',
      role: '在每个池化窗口内取平均值，相比 MaxPool 更平滑，对噪声更鲁棒。',
      uses: 'GoogLeNet 等网络中使用；全局平均池化常用于替代全连接层以减少参数。'
    },
    adaptiveavgpool2d: {
      summary: '自适应平均池化，输出尺寸固定无需计算步长。',
      role: '无论输入大小如何，自动调整池化窗口使输出恰好为指定尺寸（如 1×1）。',
      uses: '全局平均池化（output_size=1）替代全连接层；将任意尺寸特征图统一为固定尺寸。'
    },
    batchnorm1d: {
      summary: '一维批归一化，稳定全连接层训练。',
      role: '在每个 mini-batch 内对特征维度做均值/方差归一化，加速收敛、缓解 Internal Covariate Shift。',
      uses: '全连接层之后、激活之前；RNN/序列模型中对特征维度归一化。'
    },
    batchnorm2d: {
      summary: '二维批归一化，CNN 训练的标配。',
      role: '在每个 mini-batch 内对每个通道做空间维度的均值/方差归一化，使特征分布稳定。',
      uses: '几乎所有 CNN（ResNet、VGG、MobileNet）中卷积之后、激活之前的标准做法。'
    },
    layernorm: {
      summary: '层归一化，按样本归一化不受 batch 大小影响。',
      role: '对单个样本的所有特征做归一化（而非跨 batch），独立于 batch 大小，适合变长序列。',
      uses: 'Transformer / BERT / GPT 等注意力模型；RNN/LSTM 序列建模；小 batch 场景。'
    },
    groupnorm: {
      summary: '组归一化，将通道分组后归一化。',
      role: '将通道分成若干组，每组内做均值/方差归一化；介于 BatchNorm 与 LayerNorm 之间。',
      uses: '小 batch 场景（如目标检测、分割任务）；视频生成模型（如 Stable Diffusion 的 U-Net）。'
    },
    linear: {
      summary: '全连接层，对最后一维做仿射变换。',
      role: 'y = xW^T + b，将输入特征从 in_features 维映射到 out_features 维，是分类器/回归头的核心。',
      uses: '模型最后的分类头；特征维度变换；MLP 的基本单元；将展平后的特征映射到类别概率。'
    },
    relu: {
      summary: 'ReLU 激活函数，引入非线性且计算高效。',
      role: 'f(x) = max(0, x)，负值置零、正值保留，缓解梯度消失、计算极快。',
      uses: '几乎所有现代 CNN 的默认激活；卷积/全连接之后的标准选择。'
    },
    leakyrelu: {
      summary: '带泄漏的 ReLU，负值保留微小斜率。',
      role: 'f(x) = x if x>0 else negative_slope·x，避免 ReLU 的"神经元死亡"问题。',
      uses: 'GAN（如 DCGAN、StyleGAN）；ReLU 出现死神经元时的替代方案。'
    },
    gelu: {
      summary: 'GELU 激活函数，平滑的 ReLU 变体。',
      role: '高斯误差线性单元，f(x) = x·Φ(x)，在 0 附近更平滑，理论性质更优。',
      uses: 'Transformer / BERT / GPT 等注意力模型；现代预训练语言模型的标配激活。'
    },
    sigmoid: {
      summary: 'Sigmoid 激活，将输出压缩到 (0,1)。',
      role: 'f(x) = 1/(1+e^{-x})，输出可解释为概率，但深层网络中易梯度消失。',
      uses: '二分类输出层；多标签分类；门控机制（LSTM/GRU 的门）；早期神经网络的隐藏层激活。'
    },
    tanh: {
      summary: 'Tanh 激活，输出范围 (-1,1) 零中心化。',
      role: 'f(x) = (e^x - e^{-x})/(e^x + e^{-x})，相比 Sigmoid 输出零中心，收敛更快。',
      uses: 'LSTM/GRU 的隐藏状态激活；强化学习策略输出；需要对称输出的场景。'
    },
    softmax: {
      summary: 'Softmax 激活，将 logits 转为概率分布。',
      role: '沿指定维度做指数归一化，使输出和为 1，可解释为各类别概率。',
      uses: '多分类任务的输出层；注意力机制中的权重归一化。'
    },
    multiheadattention: {
      summary: '多头自注意力，Transformer 的核心组件。',
      role: '将输入投影到多个 Query/Key/Value 头并行计算注意力，再拼接投影；能捕获不同子空间的依赖关系。',
      uses: 'Transformer Encoder/Decoder；BERT/GPT 等大语言模型；ViT 图像分类；跨模态融合。'
    },
    lstm: {
      summary: 'LSTM 长短期记忆网络，解决长序列梯度问题。',
      role: '通过输入门、遗忘门、输出门控制信息流，有选择地保留/遗忘信息，缓解 RNN 梯度消失。',
      uses: '机器翻译（早期 Seq2Seq）；语音识别；时间序列预测；长文本建模（Transformer 之前的主流方案）。'
    },
    gru: {
      summary: 'GRU 门控循环单元，LSTM 的轻量替代。',
      role: '只有重置门和更新门，参数比 LSTM 少，训练更快，多数任务性能接近。',
      uses: '资源受限的序列建模；实时语音/文本处理；LSTM 的轻量替代方案。'
    },
    dropout: {
      summary: 'Dropout 随机失活，防止过拟合。',
      role: '训练时以概率 p 随机将神经元置零，迫使网络学习冗余特征；推理时自动关闭。',
      uses: '全连接层之后（p=0.5 常见）；CNN 的卷积之间（p=0.1~0.3）；Transformer 的注意力与 FFN 中。'
    },
    flatten: {
      summary: '展平操作，将多维张量拉成一维。',
      role: '把 [N, C, H, W] 展平为 [N, C*H*W]，用于在卷积特征图与全连接层之间衔接。',
      uses: 'CNN 分类头之前（卷积特征 → 全连接）；任何需要将多维特征送入 Linear 的位置。'
    },
    reshape: {
      summary: '形状变换，灵活调整张量维度。',
      role: '按指定目标形状重塑张量（元素总数不变），支持 -1 自动推断。',
      uses: '将展平的特征还原为多维；调整张量以适配不同模块的输入要求；多头注意力的头拆分。'
    },
    adaptiveavgpool: {
      summary: '自适应平均池化（通用），输出尺寸固定。',
      role: '同 AdaptiveAvgPool2d，自动计算池化窗口使输出为指定尺寸，无需手动计算 kernel/stride。',
      uses: '全局池化替代全连接；统一不同输入尺寸为固定输出。'
    },
    concat: {
      summary: '拼接操作，沿指定维度合并多个张量。',
      role: '将两个输入沿指定维度拼接，总维度增大，常用于特征融合。',
      uses: 'Inception 网络的多分支融合；DenseNet 的密集连接；多模态特征拼接（图像+文本）。'
    },
    add: {
      summary: '逐元素相加，实现残差跳连。',
      role: '将两个形状相同的张量逐元素相加，是 ResNet 残差连接的核心操作。',
      uses: 'ResNet 残差块（y = F(x) + x）；Transformer 的残差连接；任何需要跳连/特征叠加的场景。'
    },
    output: {
      summary: '模型输出标记，标识计算图的终点。',
      role: '标记模型的最终输出，代码生成时对应 forward 的 return 语句；一个模型至少需要一个 Output。',
      uses: '分类 logits 输出；回归值输出；多任务学习的多个输出节点。'
    },
    convtranspose1d: {
      summary: '一维转置卷积，用于上采样序列。',
      role: '卷积的逆操作，将低维特征上采样为高维，常用于生成模型中放大序列长度。',
      uses: '一维生成模型；序列到序列任务的上采样；时序信号重建。'
    },
    convtranspose2d: {
      summary: '二维转置卷积，图像上采样的经典方法。',
      role: '通过可学习的上采样核放大特征图空间尺寸，是 GAN 生成器与分割解码器的核心组件。',
      uses: 'DCGAN/StyleGAN 生成器；U-Net 解码器；图像分割的上采样路径；图像超分辨率。'
    },
    convtranspose3d: {
      summary: '三维转置卷积，体数据上采样。',
      role: '在 (D,H,W) 三个空间维度上同时上采样，参数量与计算量大。',
      uses: '3D U-Net 医学影像分割；视频生成；体素生成模型。'
    },
    maxpool1d: {
      summary: '一维最大池化，下采样序列。',
      role: '在序列方向滑动取最大值，降低序列长度、提取显著特征。',
      uses: '一维 CNN 文本分类；时序信号降采样。'
    },
    maxpool3d: {
      summary: '三维最大池化，体数据下采样。',
      role: '在 (D,H,W) 三个维度上取最大值，降低立体特征图尺寸。',
      uses: '3D 医学影像；视频动作识别的时空下采样。'
    },
    avgpool1d: {
      summary: '一维平均池化，平滑下采样序列。',
      role: '在序列方向取平均值，相比 MaxPool 更平滑。',
      uses: '一维信号平滑；序列模型的下采样。'
    },
    avgpool3d: {
      summary: '三维平均池化，体数据平滑下采样。',
      role: '在 (D,H,W) 上取平均，降低立体特征图尺寸且更平滑。',
      uses: '3D 医学影像；视频特征下采样。'
    },
    adaptivemaxpool2d: {
      summary: '自适应最大池化，输出尺寸固定。',
      role: '自动调整池化窗口使输出为指定尺寸，保留最显著特征。',
      uses: '全局最大池化（output_size=1）提取最显著特征；统一不同尺寸输入。'
    },
    adaptiveavgpool1d: {
      summary: '一维自适应平均池化。',
      role: '自动计算窗口使序列输出为指定长度。',
      uses: '将变长序列编码为固定长度向量；文本分类的句向量提取。'
    },
    adaptiveavgpool3d: {
      summary: '三维自适应平均池化。',
      role: '自动调整窗口使体数据输出为指定尺寸。',
      uses: '3D 特征的全局池化；统一不同尺寸的体数据输出。'
    },
    batchnorm3d: {
      summary: '三维批归一化，3D CNN 的标配。',
      role: '对 5D 输入 (N,C,D,H,W) 的每个通道做空间维归一化，稳定 3D 训练。',
      uses: '3D U-Net；视频分类网络（如 I3D、C3D）。'
    },
    instancenorm1d: {
      summary: '一维实例归一化，按样本归一化。',
      role: '对每个样本的每个通道独立归一化，风格迁移中常用。',
      uses: '风格迁移；图像生成；不依赖 batch 的归一化场景。'
    },
    instancenorm2d: {
      summary: '二维实例归一化，生成模型常用。',
      role: '对每张特征图的每个通道独立归一化，消除 batch 依赖，适合图像生成。',
      uses: '风格迁移（AdaIN）；GAN 生成器；图像翻译任务。'
    },
    instancenorm3d: {
      summary: '三维实例归一化。',
      role: '对 5D 输入的每个样本每个通道独立归一化。',
      uses: '3D 医学影像生成；视频风格迁移。'
    },
    syncbatchnorm: {
      summary: '同步批归一化，多卡训练专用。',
      role: '跨多 GPU 同步统计量，使 batch 大小 = 单卡 batch × 卡数，提升 BN 效果。',
      uses: '多卡分布式训练；大 batch 训练（如 Detectron2、BigGAN）。'
    },
    elu: {
      summary: 'ELU 激活，负值平滑的 ReLU 变体。',
      role: 'f(x)=x if x>0 else α(e^x-1)，负值区域平滑趋近 -α，输出零中心化。',
      uses: '需要零中心输出的网络；ReLU 的替代选择；深度 CNN 中减少偏置偏移。'
    },
    prelu: {
      summary: 'PReLU 参数化 ReLU，负值斜率可学习。',
      role: '负值斜率作为可学习参数自动优化，避免手动调参。',
      uses: 'ResNet 等深度 CNN；需要自适应激活的模型；人脸识别（DeepID）。'
    },
    relu6: {
      summary: 'ReLU6，限制最大值为 6。',
      role: 'f(x)=min(max(0,x),6)，防止激活值过大，利于量化。',
      uses: 'MobileNet 系列；嵌入式与移动端网络；量化友好的模型。'
    },
    selu: {
      summary: 'SELU 自归一化激活函数。',
      role: '指数线性单元的缩放变体，自动使网络输出均值0方差1，无需归一化层。',
      uses: '自归一化网络（SNN）；深度网络中替代 BN+ReLU。'
    },
    hardswish: {
      summary: 'Hardswish，Swish 的计算高效近似。',
      role: 'f(x)=x·ReLU6(x+3)/6，相比 Swish 计算更快，MobileNetV3 中提出。',
      uses: 'MobileNetV3；EfficientNet；移动端高效网络。'
    },
    hardsigmoid: {
      summary: 'Hardsigmoid，Sigmoid 的分段近似。',
      role: 'f(x)=clamp(x/6+0.5,0,1)，计算极快，量化友好。',
      uses: 'MobileNetV3 的 SE 模块；门控机制；移动端网络。'
    },
    hardtanh: {
      summary: 'Hardtanh，Tanh 的分段近似。',
      role: 'f(x)=clamp(x,min_val,max_val)，计算高效，梯度稳定。',
      uses: '量化模型；ReLU 之前的限幅；嵌入式网络。'
    },
    logsoftmax: {
      summary: 'LogSoftmax，对数概率输出。',
      role: '计算 softmax 的对数，数值稳定，常配合 NLLLoss 使用。',
      uses: '分类输出层配合 NLLLoss；强化学习的策略梯度；需要 log 概率的场景。'
    },
    softmin: {
      summary: 'Softmin，将小值赋予高权重。',
      role: '对负 softmax 归一化，使较小值获得较大概率，与 Softmax 互补。',
      uses: '需要强调小值的场景；注意力权重的反向计算。'
    },
    rnn: {
      summary: 'RNN 基础循环层，处理序列依赖。',
      role: '按时间步递推计算隐藏状态，捕获序列信息，但易梯度消失。',
      uses: '简单序列建模；基准模型；教学演示；短序列任务。'
    },
    rnncell: {
      summary: 'RNN 单步单元，手动控制时间步。',
      role: '只计算一个时间步，需在 forward 中循环调用，灵活性最高。',
      uses: '自定义序列循环；注意力机制内部；需要逐步控制的场景。'
    },
    lstmcell: {
      summary: 'LSTM 单步单元，手动控制时间步。',
      role: 'LSTM 的单步版本，需手动管理隐藏状态与细胞状态。',
      uses: '自定义序列循环；混合架构；需要精细控制 LSTM 计算的场景。'
    },
    grucell: {
      summary: 'GRU 单步单元，手动控制时间步。',
      role: 'GRU 的单步版本，需手动管理隐藏状态。',
      uses: '自定义序列循环；注意力与 RNN 混合架构；需要逐步控制的场景。'
    },
    transformerencoderlayer: {
      summary: 'Transformer 编码器单层。',
      role: '自注意力 + 前馈网络 + 残差 + LayerNorm，是 Transformer 编码器的基本单元。',
      uses: 'BERT 等编码器模型；ViT；单层 Transformer 编码；特征提取。'
    },
    transformerdecoderlayer: {
      summary: 'Transformer 解码器单层。',
      role: '自注意力 + 交叉注意力 + 前馈网络，含对编码器输出的交叉注意力。',
      uses: 'Seq2Seq 解码器；机器翻译解码；多模态解码。'
    },
    transformerencoder: {
      summary: 'Transformer 编码器，多层堆叠。',
      role: '将多个 TransformerEncoderLayer 堆叠，提取序列的深层表示。',
      uses: 'BERT/GPT 编码器；ViT 的主干；文本分类的特征提取器。'
    },
    transformerdecoder: {
      summary: 'Transformer 解码器，多层堆叠。',
      role: '将多个 TransformerDecoderLayer 堆叠，自回归解码。',
      uses: '机器翻译解码器；GPT 风格模型；文本生成。'
    },
    transformer: {
      summary: '完整 Transformer，编码器+解码器。',
      role: '原始 Transformer 架构，含编码器与解码器，用于 Seq2Seq 任务。',
      uses: '机器翻译；文本摘要；端到端 Seq2Seq 任务。'
    },
    embedding: {
      summary: 'Embedding 嵌入层，将离散 ID 映射为稠密向量。',
      role: '查表操作，将词/类别索引映射为可学习的低维向量，是 NLP 的基础。',
      uses: 'NLP 词嵌入；类别特征编码；推荐系统 ID 嵌入；图节点嵌入。'
    },
    embeddingbag: {
      summary: 'EmbeddingBag，批量嵌入聚合。',
      role: '对变长序列的嵌入做聚合（sum/mean/max），无需填充，效率高于 Embedding+池化。',
      uses: 'NLP 文本分类（FastText）；变长序列的句向量；推荐系统多值特征。'
    },
    upsample: {
      summary: 'Upsample 上采样，放大特征图尺寸。',
      role: '通过插值（最近邻/双线性等）放大特征图，无可学习参数。',
      uses: 'U-Net 解码器；分割网络的上采样；生成模型；超分辨率。'
    },
    pixelshuffle: {
      summary: 'PixelShuffle，通道转空间的上采样。',
      role: '将通道维重组为空间维，实现高效上采样，常配合卷积使用。',
      uses: '超分辨率（ESPCN、SRResNet）；实时图像生成；高效上采样。'
    },
    pixelunshuffle: {
      summary: 'PixelUnshuffle，空间转通道的下采样。',
      role: 'PixelShuffle 的逆操作，将空间维重组为通道维，无损下采样。',
      uses: '下采样而不丢失信息；与 PixelShuffle 配合；高效特征变换。'
    },
    dropout2d: {
      summary: 'Dropout2d，整通道随机失活。',
      role: '随机将整个通道置零（而非单个元素），更适合 CNN 的空间相关性。',
      uses: 'CNN 卷积之后；防止通道间共适应；正则化卷积特征。'
    },
    alphadropout: {
      summary: 'AlphaDropout，SELU 专用 Dropout。',
      role: '保持 SELU 自归一化性质的 Dropout 变体，失活后仍维持均值方差。',
      uses: '配合 SELU 的自归一化网络；需要保持激活分布的 Dropout。'
    },
    bilinear: {
      summary: 'Bilinear 双线性层，两输入的二次映射。',
      role: 'y = x1·W·x2 + b，对两个输入做双线性变换，捕获两者交互。',
      uses: '多模态融合；注意力相似度计算；推荐系统的特征交叉。'
    },
    identity: {
      summary: 'Identity 占位层，不改变输入。',
      role: '直接返回输入，常用于占位、分支对齐或调试。',
      uses: '分支对齐；条件网络中的占位；调试时替换模块。'
    },
    lazylinear: {
      summary: 'LazyLinear 延迟初始化全连接层。',
      role: 'in_features 在首次前向时自动推断，无需手动指定，方便实验。',
      uses: '不确定输入维度时；快速搭建原型；自适应输入的全连接层。'
    }
  };

  // 将描述合并到模块定义
  Object.keys(DESCRIPTIONS).forEach(type => {
    if (DEFS[type]) DEFS[type].desc = DESCRIPTIONS[type];
  });

  window.BLOCK_DEFS = DEFS;
  window.BLOCK_CATEGORIES = CATEGORIES;
  window.BLOCK_VAR_PREFIX = VAR_PREFIX;
  window.BLOCK_EXPECTED_RANK = EXPECTED_RANK;
  window.BLOCK_HELPERS = { parseShapeText, lastDim, convDim, poolDim };
})();

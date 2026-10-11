/* Chhat address parser - pure JS port of chhat/parser/infer_numpy.py.
 * BiLSTM token classifier. Weights loaded from parser_weights.bin +
 * parser_manifest.json. Produces IDENTICAL labels to the torch model.
 */
"use strict";

let W = null; // weight views
let META = null;

function matvec(mat, vec, rows, cols) {
  // mat: Float32Array rows*cols row-major, vec: Float32Array cols -> out rows
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    const off = r * cols;
    for (let c = 0; c < cols; c++) s += mat[off + c] * vec[c];
    out[r] = s;
  }
  return out;
}
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function lstmForward(x, T, inDim, wIh, wHh, bIh, bHh) {
  // x: Float32Array T*inDim. Gate order i,f,g,o. Returns Float32Array T*hidden.
  // wIh is (4*hDim, inDim), wHh is (4*hDim, hDim)
  const hDim = Math.round(wIh.length / (4 * inDim));
  const h = new Float32Array(hDim);
  const c = new Float32Array(hDim);
  const out = new Float32Array(T * hDim);
  const gates = new Float32Array(4 * hDim);
  for (let t = 0; t < T; t++) {
    const xv = x.subarray(t * inDim, (t + 1) * inDim);
    const gi = matvec(wIh, xv, 4 * hDim, inDim);
    const gh = matvec(wHh, h, 4 * hDim, hDim);
    for (let i = 0; i < 4 * hDim; i++) gates[i] = gi[i] + bIh[i] + gh[i] + bHh[i];
    for (let i = 0; i < hDim; i++) {
      const ig = sigmoid(gates[i]);
      const fg = sigmoid(gates[hDim + i]);
      const gg = Math.tanh(gates[2 * hDim + i]);
      const og = sigmoid(gates[3 * hDim + i]);
      c[i] = fg * c[i] + ig * gg;
      h[i] = og * Math.tanh(c[i]);
      out[t * hDim + i] = h[i];
    }
  }
  return out;
}

async function loadParser(baseUrl) {
  const man = await (await fetch(baseUrl + "parser_manifest.json")).json();
  META = man;
  const buf = await (await fetch(baseUrl + "parser_weights.bin")).arrayBuffer();
  W = {};
  for (const a of man.weights) {
    W[a.name] = new Float32Array(buf, a.offset, a.shape.reduce((x, y) => x * y, 1));
  }
  W.shapes = {};
  for (const a of man.weights) W.shapes[a.name] = a.shape;
}

function tokenize(text) {
  return text.replace(/,/g, " ,").trim().split(/\s+/).filter(Boolean);
}

function spansOf(labels) {
  const spans = [];
  let start = null, cur = null;
  const close = (end) => { if (start !== null && cur !== null) spans.push([start, end, cur]); };
  labels.forEach((lab, i) => {
    if (lab.startsWith("B-")) { close(i); start = i; cur = lab.slice(2); }
    else if (lab.startsWith("I-") && start !== null && cur === lab.slice(2)) { /* continue */ }
    else { close(i); start = null; cur = null; }
  });
  close(labels.length);
  return spans;
}

const HOUSE_PRE = /^(house#?|h\.?no\.?|h#|plot#?|flat#?|shop#?|p#)\s*/i;
const STREET_PRE = /^(street#?|st\.?\s*no\.?|st#?|gali#?|rd#?|no\.?)\s*/i;
const EDGE_JUNK = /^[#.\s]+|[#.\s]+$/g;
function clean(value, kind) {
  let v = value.trim();
  if (kind === "house") v = v.replace(HOUSE_PRE, "");
  else if (kind === "street") v = v.replace(STREET_PRE, "");
  return v.replace(EDGE_JUNK, "");
}

function parseAddress(text) {
  if (!W) throw new Error("parser not loaded");
  const tokens = tokenize(text);
  const T = tokens.length;
  const unk = META.vocab["<unk>"];
  const E = W.shapes["emb"][1];
  // embed
  const x0 = new Float32Array(T * E);
  for (let t = 0; t < T; t++) {
    const id = META.vocab[tokens[t]] !== undefined ? META.vocab[tokens[t]] : unk;
    x0.set(W["emb"].subarray(id * E, (id + 1) * E), t * E);
  }
  // 2-layer BiLSTM
  let x = x0, inDim = E;
  for (let l = 0; l < 2; l++) {
    const hDim = W.shapes[`lstm_l${l}_weight_hh`][1]; // (4*hDim, hDim)
    const fwd = lstmForward(x, T, inDim,
      W[`lstm_l${l}_weight_ih`], W[`lstm_l${l}_weight_hh`],
      W[`lstm_l${l}_bias_ih`], W[`lstm_l${l}_bias_hh`]);
    // reverse
    const xr = new Float32Array(T * inDim);
    for (let t = 0; t < T; t++) xr.set(x.subarray((T - 1 - t) * inDim, (T - t) * inDim), t * inDim);
    const revR = lstmForward(xr, T, inDim,
      W[`lstm_l${l}_reverse_weight_ih`], W[`lstm_l${l}_reverse_weight_hh`],
      W[`lstm_l${l}_reverse_bias_ih`], W[`lstm_l${l}_reverse_bias_hh`]);
    const rev = new Float32Array(T * hDim);
    for (let t = 0; t < T; t++) rev.set(revR.subarray((T - 1 - t) * hDim, (T - t) * hDim), t * hDim);
    const cat = new Float32Array(T * 2 * hDim);
    for (let t = 0; t < T; t++) {
      cat.set(fwd.subarray(t * hDim, (t + 1) * hDim), t * 2 * hDim);
      cat.set(rev.subarray(t * hDim, (t + 1) * hDim), t * 2 * hDim + hDim);
    }
    x = cat; inDim = 2 * hDim;
  }
  // classifier + argmax
  const nL = META.labels.length;
  const labels = [];
  for (let t = 0; t < T; t++) {
    const xt = x.subarray(t * inDim, (t + 1) * inDim);
    const sc = matvec(W["cls_w"], xt, nL, inDim);
    let best = 0;
    for (let i = 1; i < nL; i++) if (sc[i] + W["cls_b"][i] > sc[best] + W["cls_b"][best]) best = i;
    labels.push(META.labels[best]);
  }
  const fields = {};
  for (const [s, e, tag] of spansOf(labels)) {
    const k = tag.toLowerCase();
    fields[k] = (fields[k] ? fields[k] + " " : "") + tokens.slice(s, e).join(" ");
  }
  return {
    text, tokens, labels,
    house: clean(fields.house || "", "house"),
    street: clean(fields.street || "", "street"),
    block: clean(fields.block || "", "block"),
    colony: (fields.colony || "").trim(),
    city: (fields.city || "").trim(),
  };
}

// node test export
if (typeof module !== "undefined") module.exports = { loadParser, parseAddress, tokenize };

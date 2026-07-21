#!/usr/bin/env python3
import json, sys, os, numpy as np, onnxruntime
try: from tokenizers import Tokenizer
except: Tokenizer = None
class Embedder:
    def __init__(self, model_dir):
        mp = os.path.join(model_dir, "model.onnx"); tp = os.path.join(model_dir, "tokenizer.json")
        opts = onnxruntime.SessionOptions(); opts.intra_op_num_threads = 2
        opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(mp, opts)
        self.tokenizer = Tokenizer.from_file(tp)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
        self.tokenizer.enable_truncation(max_length=128)
        nms = [i.name for i in self.session.get_inputs()]
        self.dim = self.session.get_outputs()[0].shape[-1] or 384
        self.has_tt = any("token_type" in n or "segment" in n for n in nms)
        print(f"Model loaded dim={self.dim}", file=sys.stderr, flush=True)
    def embed(self, text):
        enc = self.tokenizer.encode(text)
        inp = {"input_ids": np.array([enc.ids], dtype=np.int64), "attention_mask": np.array([enc.attention_mask], dtype=np.int64)}
        if self.has_tt: inp["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)
        out = self.session.run(None, inp)[0]
        mask = np.expand_dims(inp["attention_mask"].astype(np.float32), axis=-1)
        emb = (out * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
        norm = np.linalg.norm(emb)
        if norm > 0: emb = emb / norm
        return emb[0].tolist()

def main():
    md = os.environ.get("NYAWA_MODEL_DIR", sys.argv[1] if len(sys.argv) > 1 else ".")
    emb = Embedder(md)
    print("READY", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip();
        if not line: continue
        try: req = json.loads(line)
        except: continue
        rid, method, params = req.get("id"), req.get("method", ""), req.get("params", {})
        if method == "embed":
            try: resp = {"jsonrpc": "2.0", "id": rid, "result": {"embedding": emb.embed(params.get("text", "")), "dim": emb.dim}}
            except Exception as e: resp = {"jsonrpc": "2.0", "id": rid, "error": {"code": -1, "message": str(e)}}
        else: resp = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Unknown: {method}"}}
        sys.stdout.write(json.dumps(resp) + "\n"); sys.stdout.flush()

if __name__ == "__main__": main()

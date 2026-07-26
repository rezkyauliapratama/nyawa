#!/usr/bin/env python3
"""Nyawa BGE Embedder Server — ONNX Runtime, no PyTorch."""
import json, sys, os, time
import numpy as np
import onnxruntime
try: from tokenizers import Tokenizer
except ImportError: Tokenizer = None

MODEL_DIR = os.environ.get("NYAWA_MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))

class BgeEmbedder:
	def __init__(self, model_dir=MODEL_DIR):
		self.model_dir = model_dir; self.session = None; self.tokenizer = None; self.dim = 384
		self._load()
	def _load(self):
		model_path = os.path.join(self.model_dir, "model.onnx"); tok_path = os.path.join(self.model_dir, "tokenizer.json")
		if not os.path.exists(model_path): raise RuntimeError(f"Model not found: {model_path}")
		if not os.path.exists(tok_path): raise RuntimeError(f"Tokenizer not found: {tok_path}")
		opts = onnxruntime.SessionOptions(); opts.intra_op_num_threads = 2; opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
		self.session = onnxruntime.InferenceSession(model_path, opts)
		self.tokenizer = Tokenizer.from_file(tok_path)
		self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=128)
		self.tokenizer.enable_truncation(max_length=128)
		self.dim = self.session.get_outputs()[0].shape[-1] or 384
		print(f"Model loaded: dim={self.dim}", file=sys.stderr)
	def embed(self, text):
		encoded = self.tokenizer.encode(text)
		input_ids = np.array([encoded.ids], dtype=np.int64)
		attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
		onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
		outputs = self.session.run(None, onnx_inputs)
		embedding = outputs[0]; mask = np.expand_dims(attention_mask.astype(np.float32), axis=-1)
		embedding = (embedding * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
		norm = np.linalg.norm(embedding)
		if norm > 0: embedding = embedding / norm
		return embedding[0].tolist()

def main():
	model_dir = os.environ.get("NYAWA_MODEL_DIR", None)
	if len(sys.argv) > 1 and sys.argv[1] != "serve": model_dir = sys.argv[1]
	if model_dir: global MODEL_DIR; MODEL_DIR = model_dir
	embedder = BgeEmbedder(model_dir=os.environ.get("NYAWA_MODEL_DIR", MODEL_DIR))
	print("READY", file=sys.stderr, flush=True)
	for line in sys.stdin:
		line = line.strip();
		if not line: continue
		try: req = json.loads(line)
		except json.JSONDecodeError: continue
		req_id = req.get("id"); method = req.get("method", ""); params = req.get("params", {})
		if method == "embed":
			text = params.get("text", "")
			try:
				emb = embedder.embed(text)
				response = {"jsonrpc": "2.0", "id": req_id, "result": {"embedding": emb, "dim": embedder.dim}}
			except Exception as e:
				response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -1, "message": str(e)}}
		else:
			response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}
		sys.stdout.write(json.dumps(response) + "\n"); sys.stdout.flush()

if __name__ == "__main__":
	main()

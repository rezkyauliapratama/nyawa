#!/usr/bin/env python3
"""Single-process BGE server using only stdin/stdout JSONL."""
import json, sys, os
import numpy as np
import onnxruntime
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("NYAWA_MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))

class BgeEmbedder:
	def __init__(self, model_dir=MODEL_DIR):
		self.model_dir = model_dir; self.session = None; self.tokenizer = None; self.dim = 384
		self._load()
	def _load(self):
		model_path = os.path.join(self.model_dir, "model.onnx"); tok_path = os.path.join(self.model_dir, "tokenizer.json")
		if not os.path.isfile(model_path): raise RuntimeError(f"Model not found: {model_path}")
		if not os.path.isfile(tok_path): raise RuntimeError(f"Tokenizer not found: {tok_path}")
		opts = onnxruntime.SessionOptions(); opts.intra_op_num_threads = 2; opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
		self.session = onnxruntime.InferenceSession(model_path, opts)
		self.input_names = {i.name for i in self.session.get_inputs()}
		self.tokenizer = Tokenizer.from_file(tok_path)
		self.tokenizer.enable_truncation(max_length=256)
		self.dim = self.session.get_outputs()[0].shape[-1] or 384
		print(f"Model loaded: dim={self.dim}", file=sys.stderr, flush=True)
	def embed(self, text):
		encoded = self.tokenizer.encode(text)
		input_ids = np.array([encoded.ids], dtype=np.int64)
		attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
		onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
		if "token_type_ids" in self.input_names:
			onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)
		outputs = self.session.run(None, onnx_inputs)
		embedding = outputs[0]; mask = np.expand_dims(attention_mask.astype(np.float32), axis=-1)
		embedding = (embedding * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
		norm = np.linalg.norm(embedding)
		if norm > 0: embedding = embedding / norm
		return embedding[0].tolist()


def handle_request(req, embedder):
	"""Process one JSON-RPC request dict and return the response dict."""
	req_id = req.get("id"); method = req.get("method", ""); params = req.get("params", {})
	if method == "embed":
		try:
			emb = embedder.embed(params.get("text", ""))
			return {"jsonrpc": "2.0", "id": req_id, "result": {"embedding": emb, "dim": embedder.dim}}
		except Exception as e:
			return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -1, "message": str(e)}}
	return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}


def main():
	embedder = BgeEmbedder(MODEL_DIR)
	print(f"READY dim={embedder.dim}", file=sys.stderr, flush=True)
	for line in sys.stdin:
		line = line.strip()
		if not line: continue
		try:
			resp = handle_request(json.loads(line), embedder)
		except json.JSONDecodeError as e:
			resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
		sys.stdout.write(json.dumps(resp, separators=(",", ":")) + "\n"); sys.stdout.flush()

if __name__ == "__main__":
	main()

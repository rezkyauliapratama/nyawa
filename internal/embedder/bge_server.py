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
		self.input_names = {i.name for i in self.session.get_inputs()}
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
		text = params.get("text", "")
		try:
			emb = embedder.embed(text)
			return {"jsonrpc": "2.0", "id": req_id, "result": {"embedding": emb, "dim": embedder.dim}}
		except Exception as e:
			return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -1, "message": str(e)}}
	return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}

def serve(socket_path, embedder):
	"""Serve line-delimited JSON-RPC over a unix socket (shared embedder mode).

	Multiple nyawa processes can connect to this one socket, so a single
	BGE model stays loaded in memory for the whole stack instead of one
	process per embedder. Wire protocol is identical to stdin/stdout mode.
	"""
	import socket as socketmod
	import threading
	def handle_conn(conn):
		try:
			conn.settimeout(300)
			with conn.makefile("r") as f:
				for line in f:
					line = line.strip()
					if not line: continue
					try: req = json.loads(line)
					except json.JSONDecodeError: continue
					resp = handle_request(req, embedder)
					conn.sendall((json.dumps(resp) + "\n").encode())
		except Exception:
			pass
		finally:
			try: conn.close()
			except Exception: pass
	if os.path.exists(socket_path):
		try: os.unlink(socket_path)
		except OSError: pass
	srv = socketmod.socket(socketmod.AF_UNIX, socketmod.SOCK_STREAM)
	srv.bind(socket_path)
	try: os.chmod(socket_path, 0o777)
	except OSError: pass
	srv.listen(8)
	print(f"READY (socket {socket_path})", file=sys.stderr, flush=True)
	while True:
		conn, _ = srv.accept()
		threading.Thread(target=handle_conn, args=(conn,), daemon=True).start()

def main():
	model_dir = os.environ.get("NYAWA_MODEL_DIR", None)
	serve_mode = len(sys.argv) > 1 and sys.argv[1] == "serve"
	if len(sys.argv) > 1 and not serve_mode: model_dir = sys.argv[1]
	if model_dir: global MODEL_DIR; MODEL_DIR = model_dir
	embedder = BgeEmbedder(model_dir=os.environ.get("NYAWA_MODEL_DIR", MODEL_DIR))
	if serve_mode:
		socket_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nyawa-bge.sock"
		serve(socket_path, embedder)
		return
	print("READY", file=sys.stderr, flush=True)
	for line in sys.stdin:
		line = line.strip();
		if not line: continue
		try: req = json.loads(line)
		except json.JSONDecodeError: continue
		resp = handle_request(req, embedder)
		sys.stdout.write(json.dumps(resp) + "\n"); sys.stdout.flush()

if __name__ == "__main__":
	main()

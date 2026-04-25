# PepCLIP Phase-2 Skeleton

This is a minimal DrugCLIP-style training loop for PepCLIP. It keeps Phase-1 untouched and consumes `final_metadata.jsonl`.

Current baseline:

- receptor tower: amino-acid embedding + mean pooling + projection
- peptide tower: amino-acid embedding + mean pooling + projection
- similarity: normalized dot product
- loss: symmetric in-batch softmax
- duplicate masking: exact duplicate receptor keys and peptide sequences inside a batch
- validation: Recall@K, MRR, rank metrics

Train:

```powershell
python -m phase2.pepclip.train_1d `
  --metadata_jsonl E:\pep\phase1\runs\full_run_v3\step7\final_metadata.jsonl `
  --output_dir E:\pep\phase2\runs\pepclip_1d_smoke `
  --epochs 10 `
  --batch_size 64
```

Evaluate:

```powershell
python -m phase2.pepclip.eval_retrieval `
  --metadata_jsonl E:\pep\phase1\runs\full_run_v3\step7\final_metadata.jsonl `
  --checkpoint E:\pep\phase2\runs\pepclip_1d_smoke\checkpoint_best.pt `
  --split monitor
```

This is intentionally small. The next encoder swap should preserve the same model contract:

```text
encode_receptor(...) -> [batch, dim] normalized embedding
encode_peptide(...)  -> [batch, dim] normalized embedding
```


# PepCLIP Phase-2 Skeleton

This is a minimal DrugCLIP-style training loop for PepCLIP. It keeps Phase-1 untouched and consumes Step8 Track A exports.

Current baseline:

- receptor tower: `receptor_patch_sequence` amino-acid embedding + mean pooling + projection
- peptide tower: amino-acid embedding + mean pooling + projection
- similarity: normalized dot product
- loss: symmetric in-batch softmax
- duplicate masking: exact duplicate receptor keys and peptide sequences inside a batch
- validation: Recall@K, MRR, rank metrics
- retrieval metrics are computed in chunks with `--eval_chunk_size` to avoid full `N x N` argsort memory spikes

Optional 1D encoder:

- `--encoder_type mean_pool`: default lightweight baseline
- `--encoder_type hf_esm`: HuggingFace ESM/protein model backbone + projection head

Train from Step8 LMDB:

```powershell
python -m phase2.pepclip.train_1d `
  --train_track_a E:\pep\phase1\runs\smoke_10_step8_lmdb\track_a_main_train.lmdb `
  --valid_track_a E:\pep\phase1\runs\smoke_10_step8_lmdb\track_a_monitor.lmdb `
  --data_format lmdb `
  --output_dir E:\pep\phase2\runs\pepclip_1d_smoke `
  --epochs 10 `
  --batch_size 64
```

Train from Step8 JSONL:

```powershell
python -m phase2.pepclip.train_1d `
  --train_track_a E:\pep\phase1\runs\smoke_10_step8\track_a_main_train.jsonl `
  --valid_track_a E:\pep\phase1\runs\smoke_10_step8\track_a_monitor.jsonl `
  --data_format jsonl `
  --output_dir E:\pep\phase2\runs\pepclip_1d_smoke `
  --epochs 10 `
  --batch_size 64
```

Evaluate Step8 LMDB:

```powershell
python -m phase2.pepclip.eval_retrieval `
  --track_a E:\pep\phase1\runs\smoke_10_step8_lmdb\track_a_monitor.lmdb `
  --data_format lmdb `
  --checkpoint E:\pep\phase2\runs\pepclip_1d_smoke\checkpoint_best.pt `
  --split monitor
```

Legacy Step7 `final_metadata.jsonl` is still accepted via `--metadata_jsonl`, but Step8 Track A is the preferred Phase-2 entry point.

Train with a HuggingFace ESM-style encoder:

```powershell
python -m phase2.pepclip.train_1d `
  --train_track_a E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_a_main_train.lmdb `
  --valid_track_a E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_a_monitor.lmdb `
  --data_format lmdb `
  --output_dir E:\pep\phase2\runs\pepclip_esm1d_full_v1 `
  --encoder_type hf_esm `
  --hf_model_name_or_path facebook/esm2_t6_8M_UR50D `
  --freeze_hf_backbone `
  --epochs 10 `
  --batch_size 64 `
  --eval_chunk_size 512
```

For server runs, prefer a local model directory in `--hf_model_name_or_path` if internet access is limited.

Run a DrugCLIP-style 1D control: freeze the peptide tower and train the receptor tower to align to it.

```powershell
python -m phase2.pepclip.train_1d `
  --train_track_a E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_a_main_train.lmdb `
  --valid_track_a E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_a_monitor.lmdb `
  --data_format lmdb `
  --output_dir E:\pep\phase2\runs\pepclip_esm1d_t6_drugclip_style_full_v1 `
  --encoder_type hf_esm `
  --hf_model_name_or_path E:\pep\models\esm2_t6_8M_UR50D `
  --no-freeze_hf_backbone `
  --freeze_peptide_encoder `
  --no-freeze_receptor_encoder `
  --hf_max_length 128 `
  --epochs 10 `
  --batch_size 16 `
  --lr 1e-5 `
  --eval_chunk_size 512
```

Export embeddings:

```powershell
python -m phase2.pepclip.export_embeddings `
  --track_a E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_a_monitor.lmdb `
  --data_format lmdb `
  --checkpoint E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\checkpoint_best.pt `
  --output_dir E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_monitor `
  --batch_size 64
```

Run offline topK retrieval:

```powershell
python -m phase2.pepclip.retrieve_topk `
  --query_embeddings E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_monitor\receptor_embeddings.npy `
  --target_embeddings E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_monitor\peptide_embeddings.npy `
  --metadata_jsonl E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_monitor\metadata.jsonl `
  --output_jsonl E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_monitor\top10.jsonl `
  --top_k 10 `
  --backend auto
```

Run cross-set retrieval, for example monitor receptors against the train peptide library:

```powershell
python -m phase2.pepclip.retrieve_topk `
  --query_embeddings E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_monitor\receptor_embeddings.npy `
  --target_embeddings E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_train\peptide_embeddings.npy `
  --query_metadata_jsonl E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_monitor\metadata.jsonl `
  --target_metadata_jsonl E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_train\metadata.jsonl `
  --output_jsonl E:\pep\phase2\runs\pepclip_esm1d_t6_unfrozen_e20_v1\emb_monitor\top10_vs_train.jsonl `
  --top_k 10 `
  --backend auto `
  --no-paired
```

Train the 3D-only Uni-Mol-style baseline from Step8 Track B:

```powershell
python -m phase2.pepclip.train_3d `
  --train_track_b E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_b_main_train.lmdb `
  --valid_track_b E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_b_monitor.lmdb `
  --data_format lmdb `
  --output_dir E:\pep\phase2\runs\pepclip_3d_unimol_style_full_v1 `
  --encoder_type unimol_style `
  --element_dim 128 `
  --hidden_dim 512 `
  --num_layers 4 `
  --num_heads 8 `
  --max_receptor_atoms 256 `
  --max_peptide_atoms 192 `
  --epochs 10 `
  --batch_size 32 `
  --eval_chunk_size 512
```

Evaluate the 3D checkpoint:

```powershell
python -m phase2.pepclip.eval_retrieval_3d `
  --track_b E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_b_monitor.lmdb `
  --data_format lmdb `
  --checkpoint E:\pep\phase2\runs\pepclip_3d_unimol_style_full_v1\checkpoint_best.pt `
  --eval_chunk_size 512
```

Export 3D embeddings:

```powershell
python -m phase2.pepclip.export_embeddings_3d `
  --track_b E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_b_monitor.lmdb `
  --data_format lmdb `
  --checkpoint E:\pep\phase2\runs\pepclip_3d_unimol_style_full_v1\checkpoint_best.pt `
  --output_dir E:\pep\phase2\runs\pepclip_3d_unimol_style_full_v1\emb_monitor `
  --batch_size 128
```

The default 3D mainline is a lightweight Uni-Mol-style atom transformer. It uses heavy-atom element embeddings, pairwise distance RBF features, Transformer layers, and masked pooling. The smaller `--encoder_type radial` encoder remains useful as a fast Track B smoke baseline.

DrugCLIP-style 3D pretraining freezes the peptide/molecule tower and trains the receptor/pocket tower to align to that fixed embedding space:

```powershell
python -m phase2.pepclip.train_3d `
  --train_track_b E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_b_main_train.lmdb `
  --valid_track_b E:\pep\phase1\runs\full_run_v4\step8_stage2_lmdb\track_b_monitor.lmdb `
  --data_format lmdb `
  --output_dir E:\pep\phase2\runs\pepclip_3d_unimol_style_drugclip_full_v1 `
  --encoder_type unimol_style `
  --peptide_encoder_checkpoint E:\pep\phase2\runs\peptide_3d_encoder_pretrain\checkpoint_best.pt `
  --freeze_peptide_encoder `
  --no-freeze_receptor_encoder `
  --element_dim 128 `
  --hidden_dim 512 `
  --num_layers 4 `
  --num_heads 8 `
  --max_receptor_atoms 256 `
  --max_peptide_atoms 192 `
  --epochs 10 `
  --batch_size 16 `
  --eval_chunk_size 512
```

If no pretrained peptide/Uni-Mol checkpoint is available, freezing a random peptide tower is only a wiring check. For a real DrugCLIP-style run, initialize the peptide tower first, then freeze it.

This is intentionally small. The next encoder swap should preserve the same model contract:

```text
encode_receptor(...) -> [batch, dim] normalized embedding
encode_peptide(...)  -> [batch, dim] normalized embedding
```

# Historical research snapshot

These files preserve the October 2026 exploratory implementation and recorded evaluation. They are shared to make the methods inspectable, not as a packaged or independently reproduced release. Machine-specific absolute paths were replaced with `data/` and `.`; the original workspace is unchanged.

## Components

- `build_tracking_demo.py`: reads the public phantom video and annotations, overlays contours propagated by Lucas–Kanade optical flow, and writes the replay and timeline evidence. Labels originate from the dataset.
- `prepare_search_clips.py`: samples video windows for the retrieval experiment.
- `index_embedding_video.py`: embeds raw video windows and fixed queries with a locally stored video/text model.
- `evaluate_embedding_retrieval.py`: compares saved rankings against instrument annotations after indexing.
- `serve_embedding_demo.py`: serves the historical local interface and search endpoint; it binds to loopback and expects the original workspace layout, model and vector index.
- `retrieval-evaluation.json`, `provenance.json`, and `RESULTS.txt`: preserved experiment output. The pilot used one public phantom video and four queries; Hit@1 was 1/4 and Hit@3 was 2/4.

## Reproduction limits

The dataset, model weights, indexed vectors, thumbnails, installed environment and original folder layout are not bundled. The code references the original workspace structure and requires additional setup; it has not been validated as a standalone installation. The modified paths are portability placeholders, not a claim that the pipeline now runs unchanged. The original video and dataset are subject to the UCL dataset licence linked from the main README. No patient footage is included.

The historical results are not evidence of clinical accuracy or generalization. Dataset labels drive the visual tracking anchors; they are not independently inferred by the tracker. Search similarities are not confidence probabilities.

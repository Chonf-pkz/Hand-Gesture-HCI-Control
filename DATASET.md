# Dataset for the hand gesture project

## Source

- Dataset: **HaGRID Sample 30k 384p**, a community sample derived from the original HaGRID dataset (not the full HaGRIDv2 release).
- Sample author: Christian Mills (`cj-mills`).
- Dataset page: https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p
- Downloaded revision: `40b00da`.
- Archive: `hagrid-sample-30k-384p.zip`.
- Expected archive size: **1,009,688,861 bytes**.
- Expected SHA-256: `575efb79315ee0de1415bbbacb0ded975ad156cb7c43b1538f27a91c29134861`.
- The sample author describes 31,833 images resized to 384p.
- Original project: https://github.com/hukenovs/hagrid
- License stated by the sample author: **CC BY-SA 4.0**. Preserve attribution to the sample author and original HaGRID authors. See the saved source README and HaGRID license in `data/provenance/` for details.

## Local files

- `data/raw/hagrid-sample-30k-384p/hagrid_30k/`: 31,833 source images in 18 collection folders.
- `data/raw/hagrid-sample-30k-384p/ann_train_val/`: 18 source annotation JSON files.
- `data/raw/hagrid-sample-30k-384p/annotations_df.parquet`: the sample's annotation table.
- `data/provenance/HaGRID_SAMPLE_README.md`: the sample author's dataset card.
- `data/provenance/HaGRID_LICENSE_en_us.pdf`: license copied from the original HaGRID checkout, revision `080e18917376ec935e453cd0e599c23478c7e98f`.
- `data/provenance/huggingface-files.json`: upstream file metadata, including archive size and SHA-256.
- `data/provenance/verification.json`: local verification results and data-quality observations.
- `data/provenance/previous-hagrid-requirements.txt`: the locally modified requirements from the removed reference checkout, retained for reference; it is not the dependency specification for our new project.

Raw images and downloaded metadata are excluded from Git. Keep this document in Git as a record of the dataset source. The archive matched its upstream SHA-256 before extraction. The ZIP and the cloned reference repository have been removed.

Verification found 31,833 images with corresponding JSON annotations and nonempty `user_id` fields, representing 7,973 unique user IDs. One image has a JSON label that differs from its source folder name. See `data/provenance/verification.json` for counts and details.

## Planned use

Start with five static gesture labels: `palm`, `fist`, `peace`, `like`, and `dislike`. Keep the source annotations unchanged until we implement and verify filtering together.

**Do not infer the training label from the folder name.** Inspection found images whose folder gesture is absent from their JSON labels. For example, image `70970c12-ebf8-4021-8e85-6f9f70f9b829` is stored under `train_val_fist`, while its annotation labels the hand as `no_gesture`. Read the per-hand JSON labels and inspect ambiguous samples before training. The raw files have not been relabeled.

The negative label `no_gesture` is a **per-hand annotation**; a frame may also contain another hand making a command gesture. Do not relabel an entire image as `no_command` simply because one annotated hand has `no_gesture`.

Use `user_id` to keep people separate across train, validation, and test. Run the same MediaPipe landmark extraction at training and deployment. Supplement still images with held-out webcam videos for measuring false command events per minute and complete processing latency.

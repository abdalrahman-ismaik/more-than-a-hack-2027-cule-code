# Sources and acknowledgements

- Final model implementation and frozen checkpoint: Osama Qa'adan,
  https://github.com/OsamaQ05/barca-hack-2027,
  commit `0fe3337c1b0d861fa3f5d61630687232220ff5ad` (latest remote main checked for this release).
  Model weights, decoder, feature ordering and noise statistics are copied without modification.
- Challenge configuration, scorer, validator and public five-minute labels:
  Metrica Sports, https://github.com/metrica-sports/more-than-a-hack-2027.
  Preserve the accompanying MIT notice in LICENSE. Restricted match assets are excluded.
- External training labels and tracking: DFL / Sportec Solutions IDSSE open dataset,
  distributed under CC BY 4.0 through https://huggingface.co/datasets/pysport/idsse-data.
  Match IDs: J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9.
- Numerical and machine learning software: NumPy, pandas, LightGBM, Numba and PyTorch,
  retaining their respective upstream licenses.

Earlier team experiments used SkillCorner open data, Metrica sample data and a YOLO ball
model. They do not contribute inputs or weights to the final prediction pipeline.

## AI assistance

The original pipeline was developed with Claude as a coding assistant. Codex assisted with
review, packaging, verification and documentation.





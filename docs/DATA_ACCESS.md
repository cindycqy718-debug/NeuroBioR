# Data Access Boundary

The companion archive contains the exact aligned TEST inputs and private scoring labels
for APAVA and ADFTD. The public subdirectories intentionally contain no labels or source
subject identifiers; private manifests are used only by the local scorer.

TDBRAIN is access-controlled and is not redistributed. Researchers who are authorized
by the dataset owner may use the pinned preprocessing notebook at
`third_party/Medformer/data_preprocessing/TDBRAIN_preprocessing.ipynb`. The expected
aligned TEST contract is 960 windows, 33 channels, 256 time points, 30 input batches,
and 8 test subjects.

Do not upload raw or processed TDBRAIN signals to GitHub or a public file share unless
the dataset owner explicitly permits redistribution.


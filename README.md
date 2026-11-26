# How much do evaluation choices matter?

### Revisiting class-weighted focal-loss transfer learning for skin lesion classification on HAM10000

A re-implementation and controlled comparison built on **Le et al. (2020)**, *Transfer learning
with class-weighted and focal loss function for automatic skin cancer classification*
([arXiv:2009.05977](https://arxiv.org/abs/2009.05977)).

The HAM10000 dataset contains 10,015 dermoscopy images of only 7,470 distinct lesions — many
lesions were photographed more than once, at different zoom levels or angles. If a random
image-level split is used, two photographs of the *same* physical lesion can end up on opposite
sides of the train/test boundary. This repository measures how much that single choice changes
the reported numbers, holding the model and the training procedure fixed.

Le et al. (2020) highlighted the importance of a held-out test set. We extend this by quantifying
how strongly HAM10000 results depend on whether images of the same lesion can appear in both the
training and test sets, and by testing whether the benefits of class weighting and focal loss hold
across repeated training runs. We re-run everything ourselves under two explicit split protocols
and never re-evaluate the published numbers.

## Research questions

1. **Split protocol.** With the same model and training, how different are test results under an
   **image-level** split versus a **lesion-level** split?
2. **Imbalance techniques.** Under a lesion-level split, do class weights and focal loss still
   help once run-to-run variance is accounted for (5 seeds, mean ± SEM, Welch's t-test)?
3. **Clinical view.** How well is melanoma actually detected (sensitivity and specificity),
   compared with what overall accuracy suggests?

Every outcome is reportable: a large gap between splits, no gap, focal loss confirmed, or focal
loss within noise.

## Background in one paragraph

**Melanoma** is a cancer of the pigment-producing melanocytes. It accounts for roughly 1% of skin
cancer cases but most skin cancer deaths, because it metastasises — and it is very treatable when
caught early, which makes early detection the whole game. It is also hard to distinguish from an
ordinary benign mole. **Dermoscopy** (imaging through a magnifier with a built-in light source) is
the standard tool; HAM10000 consists of dermoscopy images. The dataset is severely **imbalanced** —
two thirds of it is the benign class `nv` — so overall accuracy is dominated by one easy class and
says very little about melanoma.

## Data

HAM10000 (Tschandl et al., 2018), 10,015 images at 600×450 px, 7 diagnostic classes. Labels are
confirmed by histopathology, confocal microscopy, follow-up or expert consensus.

| class     | meaning                        | images |
| --------- | ------------------------------ | -----: |
| `nv`    | melanocytic nevi (benign mole) |  6,705 |
| `mel`   | melanoma                       |  1,113 |
| `bkl`   | benign keratosis               |  1,099 |
| `bcc`   | basal cell carcinoma           |    514 |
| `akiec` | actinic keratosis              |    327 |
| `vasc`  | vascular lesion                |    142 |
| `df`    | dermatofibroma                 |    115 |

10,015 images, **7,470 unique `lesion_id`s**. That gap is the whole subject of RQ1.

[src/prepare_data.py](src/prepare_data.py) downloads the dataset via `kagglehub`, resizes every
image once to 224×224, and stores the result as a single `uint8` `.npy` array (~1.5 GB) plus a CSV
index. One big file instead of 10,015 small ones keeps the HPC filesystem happy and makes epochs
fast. It asserts the image count, the lesion count and the known class counts, and refuses to
continue if any disagree.

## The two splits

Both use ~20% test, then 80:20 train:validation on the remainder, stratified by diagnosis, with
the split seed fixed at 0. The six resulting CSVs are **committed to this repo** — they define the
experiment, so they should not be regenerated silently.

- **Split A — image-level.** Random stratified split over all 10,015 images, ignoring `lesion_id`.
  Images of one lesion may land in different subsets. This is the permissive protocol.
- **Split B — lesion-level.** `StratifiedGroupKFold` grouped on `lesion_id`, so every image of a
  lesion stays in one subset. Zero lesion overlap between train, validation and test is asserted,
  not assumed.

| split                    | train                     | validation                | test                      |
| ------------------------ | ------------------------- | ------------------------- | ------------------------- |
| **A** image-level  | 6,409 img / 5,275 lesions | 1,603 img / 1,523 lesions | 2,003 img / 1,885 lesions |
| **B** lesion-level | 6,409 img / 4,781 lesions | 1,602 img / 1,194 lesions | 2,004 img / 1,495 lesions |

### The cleanest evidence: seen vs unseen lesions

Comparing two splits also compares two different test sets, which muddies the picture. So in
Split A every test image is additionally tagged according to whether another image of the *same*
lesion appears in training:

- **seen** — 651 images whose lesion also appears in train
- **unseen** — 1,352 images whose lesion does not

One model, one test set, two subgroups. That isolates the effect of lesion overlap without
changing anything else.

> **Caveat — do not compare raw accuracy between these two subgroups.** They have very different
> class composition:
>
> | group  |     n | akiec |  bcc |   bkl |   df |             mel |              nv | vasc |
> | ------ | ----: | ----: | ---: | ----: | ---: | --------------: | --------------: | ---: |
> | seen   |   651 |  3.5% | 6.9% | 13.8% | 2.2% | **21.0%** | **50.8%** | 1.7% |
> | unseen | 1,352 |  3.1% | 4.3% |  9.6% | 0.7% |  **6.4%** | **74.7%** | 1.3% |
>
> This is not a sampling bug — it follows from how the dataset was collected. Lesions worth
> photographing twice are disproportionately the clinically suspicious ones, so the seen group
> carries roughly three times the melanoma share while the unseen group is three-quarters easy
> `nv`. That alone inflates accuracy on the unseen group. Only prevalence-robust metrics
> (macro-F1, balanced accuracy, per-class recall) are meaningful for this contrast.

## Model and training

ImageNet-pretrained **ResNet50** (`IMAGENET1K_V2`), all layers trainable, with the paper's head:
global average pooling → `Linear(2048, 512)` → ReLU → `Dropout(0.5)` → `Linear(512, 7)`.

Adam at lr 1e-4, batch 64, `ReduceLROnPlateau` on validation loss (factor 0.5, patience 3), max 30
epochs, early stopping after 7 epochs without improvement. The kept checkpoint is the one with the
best **validation macro-F1** — not accuracy, which would just track the majority class. bfloat16
mixed precision. Train-time augmentation: horizontal and vertical flips, rotation (20°),
`RandomResizedCrop(224, scale=(0.8, 1.0))`, and `RandomErasing` standing in for cutout.

Each run is one `(split, config, seed)` triple with training seeds 0–4. The split itself stays
fixed, so the only thing varying across seeds is training randomness.

### Loss configurations

Class weights $w_c$ are inverse class frequency on the **training** split, normalised to mean 1.
Focal loss uses $\gamma = 2$.

| config    | loss                                         |
| --------- | -------------------------------------------- |
| `full`  | focal loss, $\gamma=2$, $\alpha_t = w_c$ |
| `no_fl` | weighted cross-entropy with $w_c$          |
| `no_cw` | focal loss, $\gamma=2$, $\alpha_t = 1$   |
| `plain` | unweighted cross-entropy                     |

## Metrics and statistics

Per run, on that split's fixed test set: accuracy (for comparability with the paper), **macro-F1**,
balanced accuracy, macro one-vs-rest AUC, per-class precision/recall/F1, the confusion matrix, and
**melanoma sensitivity and specificity** one-vs-rest. For Split A, all of these are additionally
computed on the seen and unseen subgroups.

Every run writes `results.json` (config, seed, best epoch, all metrics) and `test_probs.npz`
(per-image probabilities and labels), so any further metric can be recomputed later without
retraining. [src/analyze.py](src/analyze.py) aggregates them into mean ± SEM over seeds and runs
Welch's t-tests: Split A vs B for `full`, seen vs unseen within Split A, and each ablation against
`full` under Split B.

Five seeds is a small sample. Effect sizes (Cohen's *d*) are reported next to every p-value, and
"significant" here refers only to variation across training runs on a fixed test set — not to
sampling variation over patients.

## Run table

`--task-id` indexes `RUN_TABLE` in [src/train.py](src/train.py):

| task-id | split | config | seeds | tier     |
| ------- | ----- | ------ | ----- | -------- |
| 0–4    | A     | full   | 0–4  | 1 (must) |
| 5–9    | B     | full   | 0–4  | 1 (must) |
| 10–14  | B     | no_fl  | 0–4  | 2        |
| 15–19  | B     | no_cw  | 0–4  | 2        |
| 20–24  | B     | plain  | 0–4  | 3        |

Tier 1 answers RQ1 and RQ3, Tier 2 answers RQ2, Tier 3 is the unweighted baseline.

## Layout

```
src/prepare_data.py   download, resize to 224x224, verify, save npy + index csv
src/make_splits.py    Split A (image-level) and Split B (lesion-level) + assertions
src/losses.py         focal loss, weighted CE
src/train.py          one run = (split, config, seed); --task-id maps to RUN_TABLE
src/check_env.py      GPU / compute-capability check, run before submitting
src/analyze.py        aggregate results.json, mean±SEM, Welch tests
slurm/train_array.sh  job array, one array task per run
splits/*.csv          the six fixed split files (committed: these define the experiment)
docs/                 the paper this work builds on
```

`data/`, `results/`, `report/` and `logs/` are generated and deliberately untracked.

## Deviations from Le et al. (2020)

Every point where this re-implementation differs from the paper, or where the paper does not
report enough detail to match it. "Not reported" means the value is absent from the paper and we
chose one.

**Framework and model**

| item             | paper                                                          | ours                                                            | why                                                                 |
| ---------------- | -------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------- |
| framework        | Keras (Google Colab)                                           | PyTorch + torchvision                                           | our environment                                                     |
| ResNet50 weights | ImageNet-pretrained, version unspecified                       | `ResNet50_Weights.IMAGENET1K_V2`                              | paper predates the V1/V2 distinction; recorded so it can be swapped |
| head hidden size | not reported ("fully connected → dropout → fully connected") | `Linear(2048, 512) → ReLU → Dropout(0.5) → Linear(512, 7)` | 512 is our assumption; dropout 0.5 matches the paper                |
| precision        | not reported                                                   | bfloat16 mixed precision                                        | speed, supported on our GPUs                                        |

**Loss**

| item            | paper                                                                       | ours                                                                | why                                                                        |
| --------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| focal loss γ   | not reported                                                                | 2.0                                                                 | standard value from Lin et al. (2017)                                      |
| focal loss α_t | "initialised from class ratios, then hand-tuned", final values not reported | inverse class frequency on the training split, normalised to mean 1 | hand-tuning we cannot see cannot be reproduced; computed from data instead |
| class weights   | hand-tuned                                                                  | the same fixed inverse-frequency weights, untuned                   | deliberate simplification, for reproducibility                             |

**Training schedule**

| item                 | paper                                             | ours                                                                   | why                                          |
| -------------------- | ------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------- |
| optimiser / LR       | Adam, 1e-4                                        | same                                                                   | —                                           |
| batch size           | 64                                                | same                                                                   | —                                           |
| image size           | 224×224                                          | same (resized once, bilinear, stored as uint8)                         | —                                           |
| LR schedule          | "reduce LR on plateau", parameters not reported   | on validation loss, factor 0.5, patience 3                             | our choice                                   |
| epochs               | best single model at 26 epochs                    | max 30, early stop after 7 without improvement                         | our choice                                   |
| checkpoint selection | "checkpointing on validation", metric unspecified | best validation macro-F1                                               | imbalance-aware; accuracy would track `nv` |
| augmentation         | rotation, flipping, cutout, cropping              | same families; rotation 20° and crop scale (0.8, 1.0) are our choices | exact magnitudes not reported                |

**Evaluation — this is the point of the project**

| item             | paper                                                                                                                                                                                      | ours                                                                                              | why                                                                                          |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| split            | ambiguous: the text says duplicates were removed (7,470 lesions), but the reported train+val+test sizes sum to 10,015, so we cannot tell whether a test lesion can also appear in training | two explicit protocols, Split A image-level and Split B lesion-level with zero overlap asserted   | this comparison *is* the research question; the paper's own numbers are never re-evaluated |
| repeats          | one run per ablation row                                                                                                                                                                   | 5 training seeds per (split, config), split fixed                                                 | enables mean ± SEM and Welch's t-tests rather than single-run differences                   |
| ensembling / TTA | 5-fold CV ensemble + test-time augmentation for the headline result                                                                                                                        | not done                                                                                          | out of scope; we compare single models under controlled conditions                           |
| extra metrics    | accuracy, precision, recall, F1                                                                                                                                                            | also balanced accuracy, macro OvR AUC, melanoma sensitivity/specificity, seen-vs-unseen breakdown | needed for RQ1 and RQ3                                                                       |

**Not attempted:** Grad-CAM visualisations, the segmentation experiments (U-Net / dilated masks),
and the alternative backbones (VGG16, MobileNet, EfficientNetB1).

**On split sizes.** Ours (above) do not match the paper's 6,817 / 1,704 / 1,494 — expected, since
we split all 10,015 images rather than applying the paper's ambiguous duplicate-removal procedure.

## Reproducing

Needs a CUDA GPU. Paths below assume the Hábrók HPC cluster with the repo at
`$HOME/healthcare_ai_project`, data on `/scratch/$USER/ham10000` and results on
`/scratch/$USER/ham_results`; these are set at the top of
[slurm/train_array.sh](slurm/train_array.sh).

**1. Get the code there.**

```bash
git clone https://github.com/simotekam/HAM10000-skin-lesion-classification-evaluation-choice
```

Checked 26 Nov 2026: `login1` was refusing connections on port 22; `login2`, `interactive1` and
`interactive2` were up. Use whichever answers.

**2. Build the venv on a GPU node** — it must match the GPU, not the login node. The central
PyTorch modules are built for V100/A100 and fail on the RTX Pro 6000 (compute capability 12.0), so
install your own.

```bash
srun --gres=gpu:1 --time=01:00:00 --pty /bin/bash   # interactive GPU node
unset SW_STACK_ARCH && module restore               # only if the stack looks wrong

module purge
module load Python/3.13.1-GCCcore-14.2.0
python3 -m venv $HOME/venvs/ham_venv
source $HOME/venvs/ham_venv/bin/activate
pip install --upgrade pip wheel
pip install -r $HOME/healthcare_ai_project/requirements-habrok.txt
```

**3. Check the environment**, still on the GPU node:

```bash
python $HOME/healthcare_ai_project/src/check_env.py
```

Must report compute capability `12.0` and `sm_120` in the arch list. If `sm_120` is missing the
torch wheel is wrong — reinstall with a newer CUDA build before going further. Then `exit`.

**4. Prepare the data on the LOGIN node** — compute nodes have no internet.

```bash
module load Python/3.13.1-GCCcore-14.2.0
source $HOME/venvs/ham_venv/bin/activate
cd $HOME/healthcare_ai_project

python src/prepare_data.py --out-dir /scratch/$USER/ham10000
python src/make_splits.py  --data-dir /scratch/$USER/ham10000
```

Keep only the `.npy` on `/scratch`, not 10,015 loose JPEGs — there is a file-count limit. The
`kagglehub` cache under `~/.cache/kagglehub` holds the extracted images; delete it once the `.npy`
exists if you are near quota. Note that `make_splits.py` overwrites the committed `splits/*.csv`;
with seed 0 it reproduces them exactly.

**5. Smoke-test on a GPU before submitting anything.**

```bash
srun --gres=gpu:1 --time=00:30:00 --pty /bin/bash
module load Python/3.13.1-GCCcore-14.2.0
source $HOME/venvs/ham_venv/bin/activate
cd $HOME/healthcare_ai_project
python -u src/train.py --split B --config full --seed 0 \
  --epochs 1 --max-images 700 \
  --data-dir /scratch/$USER/ham10000 \
  --output-dir /tmp/smoke
exit
```

**6. Submit the real runs.**

```bash
cd $HOME/healthcare_ai_project
mkdir -p logs
sbatch --array=0-9   slurm/train_array.sh    # Tier 1
sbatch --array=10-19 slurm/train_array.sh    # Tier 2
sbatch --array=20-24 slurm/train_array.sh    # Tier 3
```

Let Tier 1 land before adding more — there is a cap on queued+running jobs per user, and exceeding
it fails with *"Job violates accounting/QOS policy"*.

**7. Monitor.**

```bash
squeue -u $USER                      # queue state
squeue -u $USER --start              # estimated start times
tail -f logs/ham_<jobid>_0.out       # live log of one array task
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
scancel <jobid>                      # cancel whole array
scancel <jobid>_3                    # cancel one task
```

**8. Analyse**, then copy the results somewhere backed up — `/scratch` is not.

```bash
python src/analyze.py --results-dir /scratch/$USER/ham_results --out-dir report

mkdir -p /projects/$USER/ham10000
cp -r /scratch/$USER/ham_results report /projects/$USER/ham10000/
```

`$HOME` (50 GB) and `/projects` (reachable from login/interactive nodes only) are backed up. The
job script copies the 1.5 GB `.npy` to `$TMPDIR` (fast node-local disk) at the start of each task,
since it is read every epoch.

## References

- Le, D.N.T., Le, H.X., Ngo, L.T., and Ngo, H.T. (2020). Transfer learning with class-weighted and
  focal loss function for automatic skin cancer classification. arXiv:2009.05977.
- Tschandl, P., Rosendahl, C., and Kittler, H. (2018). The HAM10000 dataset, a large collection of
  multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). Focal loss for dense object
  detection. ICCV.
- He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. CVPR.
- Abhishek, K., et al. (2025). Investigating the quality of DermaMNIST and Fitzpatrick17k
  dermatological image datasets. *Scientific Data*, 12, 196.

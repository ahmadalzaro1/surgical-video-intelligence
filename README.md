# Surgical video intelligence

Exploratory research for navigating surgical video using annotation-seeded instrument tracking and video search, tested on a public **bench-top phantom**.

[Research snapshot](research_snapshot/) · [Recorded evaluation](research_snapshot/retrieval-evaluation.json) · [Dataset](https://doi.org/10.5522/04/26511049.v1) · [Companion paper](https://doi.org/10.1049/htl2.12101)

![Diagram showing the phantom video, tracking and video search workflow](research_snapshot/pipeline.svg)

**Status:** research snapshot and visual preview. The full video, model weights and source dataset are not included. No patient footage is published.

<details>
<summary>View the tracking interface</summary>

![Screenshot of the instrument-tracking research demo](demo-screenshot.png)

</details>

## What the research explores

- Published dataset annotations seed instrument contours and labels.
- Lucas–Kanade optical flow propagates contour points between one-second annotation anchors, with forward/backward consistency checks.
- An interactive timeline lets a viewer navigate instrument intervals in the recorded video.
- A separate video-only semantic-search pilot was evaluated on four fixed queries; relevant intervals appeared in the top three for two queries.

```mermaid
flowchart LR
  A[Public phantom video] --> B[Dataset annotation anchors]
  B --> C[Optical flow and point checks]
  C --> D[Replay and instrument timeline]
  A --> E[Video embedding search pilot]
```

## Interpretation

Instrument identities shown in the tracking display come from dataset labels, not autonomous recognition. These results concern a small, exploratory test and do not establish clinical validity, patient outcomes or generalization. Tracking is re-anchored once per second. The search pilot missed two of its four fixed instrument queries.

## Data and attribution

The visual demo uses the public [Endoscopic Pituitary Surgery on a High-fidelity Bench-top Phantom dataset](https://doi.org/10.5522/04/26511049.v1) by Adrito Das, Danyal Khan, Nicola Newall, John Hanrahan, Danail Stoyanov, Hani Marcus and Sophia Bano, made available under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The screenshot is an adapted visualization of that dataset and is shared under the same license. See also the dataset's [companion paper](https://doi.org/10.1049/htl2.12101).

The [historical research snapshot](research_snapshot/) includes the tracking and retrieval scripts and recorded evaluation. It is a source snapshot, not a packaged standalone release; the full replay, dataset and model weights are not included.

## New: SAM 2 and optical-flow comparison

A completed phantom-video experiment compares annotation-seeded SAM 2 with an optical-flow baseline. On 329 eligible one-second transitions, mean mask IoU was 0.784 for SAM 2 and 0.563 for the baseline; the evaluation is exploratory and does not establish clinical performance.

![Side-by-side preview of optical-flow and SAM 2 tracking on phantom video](experiments/case5-sam2-dis/assets/case5-tracking-comparison.gif)

[See the experiment, methodology, results and attribution](experiments/case5-sam2-dis/)

**Second recording — complete:** Case 11 contains 772 one-second intervals and 383 eligible paired transitions. Mean IoU was 0.683 for SAM 2 and 0.554 for DIS. As with case 5, masks and labels were supplied once per second; this does not measure autonomous detection or long-term tracking. [See the final results](experiments/case5-sam2-dis/results/case11-summary.json).

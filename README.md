# Surgical video intelligence

Exploratory work on making surgical-video material easier to inspect, using a public endoscopic pituitary surgery **bench-top phantom**. The visual demo shows an annotated replay and instrument-state timeline; it does not contain patient footage.

![Screenshot of the instrument-tracking research demo](demo-screenshot.png)

## What the demonstration shows

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

![](./assets/Grounded-SAM_logo.png)

This repository is forked from https://github.com/IDEA-Research/Grounded-Segment-Anything (commit 99ec802)


# Grounded-Segment-Anything

The **core idea** behind this project is to **combine the strengths of different models in order to build a very powerful pipeline for solving complex problems**. And it's worth mentioning that this is a workflow for combining strong expert models, where **all parts can be used separately or in combination, and can be replaced with any similar but different models (like replacing Grounding DINO with GLIP or other detectors / replacing Stable-Diffusion with ControlNet or GLIGEN/ Combining with ChatGPT)**.

- [Segment Anything](https://github.com/facebookresearch/segment-anything) is a strong segmentation model. But it needs prompts (like boxes/points) to generate masks. 
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) is a strong zero-shot detector which is capable of to generate high quality boxes and labels with free-form text. 
- The combination of `Grounding DINO + SAM` enable to **detect and segment everything at any levels** with text inputs!



# SemEval-Task5-MAMI
Implementation of our SemEval-Task5-MAMI challenge <a href="https://arxiv.org/abs/2204.03953">paper</a>.
### How to run the code
* Download the pre-trained image-caption model from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning, save the downloaded model "BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth" under `./local/imagecap`
* run `run.sh` from the beginning
### File architecture
* processed Dataset and features are saved in `./Dataset`
* trained models are saved in `./model`
* results are saved in `./results`

## Citations
```bibtex
@misc{https://doi.org/10.48550/arxiv.2204.03953,
  doi = {10.48550/ARXIV.2204.03953},
  url = {https://arxiv.org/abs/2204.03953},
  author = {Yu, Wentao and Boenninghoff, Benedikt and Roehrig, Jonas and Kolossa, Dorothea},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {RubCSG at SemEval-2022 Task 5: Ensemble learning for identifying misogynous MEMEs},
  publisher = {arXiv},
  year = {2022}
}

```

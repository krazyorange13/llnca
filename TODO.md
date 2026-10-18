## todos

### developmental

- [ ] overhaul eval2.py for eval3.py
- [x] use new embeddings2
- [ ] merge tokenizer2.py into tokenizer.py
- [ ] implement config diffs (smth | None = None)
- [ ] implement config viewing (eval2.py)
- [ ] implement config loading for past versions (smth cool or getattr/setattr)
- [ ] implement robust training data saving (loss, gradients, eval performance)
- [ ] implement grid search (fast fail, binary search exploration)
- [ ] implement eval2.py launching and resuming grid searches and training runs
- [ ] automatically launch and monitor kaggle notebooks
- [ ] ~~use ur damn google cloud credits before they run out~~

### experimental

- [ ] initialize NCA last layer to zero (test performance with grid search)
- [ ] disable alive masking (test performance with grid search)
- [ ] finally use the adversarial NCA
- [ ] keep researching GRUs
- [ ] keep researching other gradient stabilization techniques
- [ ] keep researching VNCA, HyperNCA, TextNCA, CAX, and Pre-pre-training LLMS

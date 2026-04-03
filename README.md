## llnca

large language neural cellular automata

modelling Large Language Models (LLMs) using Neural Cellular Automata (NCAs).

### thoughts

NCAs are highly adept at repairing broken images and patterns. if the NCA is trained
on an image of a lizard, and the tail is erased, the NCA can regrow the tail.
subsequently, if an NCA is trained on an image of the word "hi", and the "i" is
erased, the NCA can regrow the "i". further, if the NCA is trained on an image
of the sentence "hello world!", and the "world!" is erased, the NCA can regenerate
the "world!". by now you can probably see where this is going :&#8203;P i hypothesize
that if an NCA is trained on many, many images of text, then it will be able repair
missing words based on the surrounding words. i aim to push NCA to their limits
by training them to complete truncated texts, similar to how LLMs work.

### pipeline

1. pull random english sentences from online databases or books
2. normalize the sentences and organize them into bins by length
3. generate a truncated seed from each sentence
4. render the seeds and full sentences into images
5. populate random training pools for each bin
6. train the NCA to start with a seed image and complete the full image
7. ???
8. become a billionaire

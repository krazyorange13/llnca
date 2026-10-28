import sys

import pandas as pd
import plotly.express as px
import torch
import torch.nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from corpus import (
    LLNCACorpus,
    LLNCACorpusConfig,
    LLNCACorpusSlidingConfig,
    LLNCACorpusTruncConfig,
)
from dataset import (
    LLNCADataSampler,
    LLNCADataSamplerConfig,
    LLNCADataset,
    LLNCADatasetConfig,
)
from embedding import LLNCAEmbeddings, LLNCAEmbeddingsConfig
from main import (
    LLNCA,
    LLNCAAdvConfig,
    LLNCAConfig,
    LLNCAGenConfig,
    LLNCAOptimConfig,
    LLNCASaverConfig,
)
from nca import LLNCANCA, LLNCANCAConfig
from tokenizer import LLNCATokenizer, LLNCAVocab
from tokenizer2 import LLNCATokenizer2, LLNCATokenizer2Config, LLNCAVocab2

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("please provide a .pth checkpoint file to load from")
        sys.exit(1)

    path = sys.argv[1]

    llnca = LLNCA(
        checkpoint=torch.load(
            path,
            weights_only=False,
            map_location=torch.device("cpu"),
        )
    )

    embedding_weights = llnca.embeddings.embeddings.weight.detach().cpu().numpy()

    # pca = PCA(n_components=3)
    # results = pca.fit_transform(embedding_weights)

    tnse = TSNE(
        n_components=3,
        perplexity=15,
        random_state=42,
        learning_rate="auto",
    )
    results = tnse.fit_transform(embedding_weights)

    vocab = []

    assert llnca.tokenizer.tokenizer is not None
    vocab = [
        llnca.tokenizer.decode([i])
        for i in range(len(llnca.tokenizer.tokenizer.vocab) + 1)
    ]

    df = pd.DataFrame(results, columns=["pca1", "pca2", "pca3"])
    df["token"] = vocab
    df["mag"] = (
        torch.norm(llnca.embeddings.embeddings.weight, dim=1).detach().cpu().numpy()
    )

    fig = px.scatter_3d(
        df,
        x="pca1",
        y="pca2",
        z="pca3",
        text="token",
        color="mag",
        color_continuous_scale="viridis",
        title=f"{path} embeddings",
    )

    fig.update_traces(marker={"size": 5, "opacity": 0.8})
    fig.update_layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        scene={"xaxis_title": "pca1", "yaxis_title": "pca2", "zaxis_title": "pca3"},
    )

    fig.show()

from typing import List

import torch

import flair.datasets
from flair.data import Corpus
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
)
from flair.visual.training_curves import Plotter


def main():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # 1. get the corpus
    corpus: Corpus = flair.datasets.UD_ENGLISH()
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = "upos"

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary.idx2item)

    # initialize embeddings
    embedding_types: List[TokenEmbeddings] = [
        WordEmbeddings("glove"),
        # comment in this line to use character embeddings
        # CharacterEmbeddings(),
        # comment in these lines to use contextual string embeddings
        #
        # FlairEmbeddings('news-forward'),
        #
        # FlairEmbeddings('news-backward'),
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True,
        device=device
    )

    # initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train(
        "resources/taggers/example-ner",
        learning_rate=0.1,
        mini_batch_size=32,
        max_epochs=20,
        shuffle=False,
    )

    plotter = Plotter()
    plotter.plot_training_curves("resources/taggers/example-ner/loss.tsv")
    plotter.plot_weights("resources/taggers/example-ner/weights.txt")


if __name__ == "__main__":
    main()

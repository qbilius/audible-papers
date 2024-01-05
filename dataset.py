from pathlib import Path

import numpy as np
import torch
import tqdm


class Citations:

    def __init__(self, data_dir: Path, seed: int | None = 0):
        self.rng = np.random.default_rng(seed)
        self.lastnames = np.load(data_dir / 'lastnames.npy')

    def get_year(self) -> int:
        return self.rng.integers(1990, 2024)

    def get_page(self) -> str:
        return f'p. {self.rng.integers(0, 400)}'

    def brackets(self, p=None) -> tuple[str, str]:
        kinds = [('(', ')'), ('[', ']'), ('', '')]
        bo, bc = kinds[self.rng.choice(range(len(kinds)), p=p)]
        return bo, bc

    def text(self) -> str:
        n_authors = self.rng.choice([1, 2, 3, 4], p=[.1, .1, .1, .7])
        authors = self.rng.choice(self.lastnames, n_authors, replace=False)
        if n_authors == 1:
            authors_list = authors[0]
        elif n_authors == 2:
            authors_list = f'{authors[0]} & {authors[1]}'
        elif n_authors == 3:
            authors_list = f'{authors[0]}, {authors[1]}, & {authors[2]}'
        else:
            authors_list = f'{authors[0]} et al.'

        year = self.get_year()
        page = self.get_page()
        citation = f'{authors_list}, {year}'
        if self.rng.random() < .1:
            citation += f', {page}'

        if self.rng.random() < .5:
            citation = citation.replace('&', 'and')

        if self.rng.random() < .2:
            citation = citation.replace('.', '')

        if self.rng.random() < .2:
            citation = citation.replace(',', '')

        if self.rng.random() < .2:
            citation = citation.replace('. ', '.')

        if self.rng.random() < .2:
            citation = citation.replace(', ', ',')

        return citation

    def numbers(self) -> str:
        if self.rng.random() < .9:
            return str(self.rng.integers(1, 110))
        else:
            s, e = self.rng.choice(range(1, 110), size=2, replace=False)
            return f'{s}-{e}'

    def __call__(self) -> str:
        n_sources = self.rng.choice([1, 2, 3, 4], p=[.9, .05, .025, .025])
        if self.rng.random() < .5:
            bo, bc = self.brackets(p=[.9, .09, .01])
            citation = '; '.join([self.text() for _ in range(n_sources)])
        else:
            bo, bc = self.brackets()
            citation = ', '.join([self.numbers() for _ in range(n_sources)])
            if self.rng.random() < .5:
                citation = citation.replace(', ', ',')
        return bo + citation + bc


class Augment:

    def __init__(self,
                 citation_values: list | np.ndarray,
                 block_size: int,
                 seed: int | None = 0
                 ):
        self.rng = np.random.default_rng(seed)
        self.citation_values = list(citation_values)
        self.block_size = block_size

    def __call__(self, block: list) -> tuple[list, list]:
        n_citations = self.rng.choice([1, 2], p=[.95, .05])
        inds = self.rng.choice(range(len(block)), size=n_citations, replace=False)
        inds = sorted(inds, reverse=True)
        x = list(block)
        for idx in inds:
            citation_values = self.citation_values[self.rng.integers(len(self.citation_values))]
            citation_values = [c for c in citation_values if c != 50257]
            if self.rng.random() > .05:
                # add a space between preceeding text and citation
                citation_values = [220] + citation_values
            else:
                citation_values = citation_values

            y = [False] * len(x[:idx]) + [True] * len(citation_values) + [False] * len(x[idx:])
            x = x[:idx] + citation_values + x[idx:]

        return x[:self.block_size], y[:self.block_size]


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir: Path,
                 block_size: int,
                 stage: str,
                 ) -> None:
        super().__init__()
        self.block_size = block_size
        self.stage = stage

        data = np.array(np.memmap(data_dir / f'{stage}.bin', np.uint16, mode='r'))
        blocks = np.array_split(data, len(data) // block_size)
        citation_values = np.load(data_dir / f'{stage}_citations.npy')

        self.aug = Augment(citation_values, self.block_size)
        if stage == 'val':
            # aug = Augment(citation_values, self.block_size)
            self.samples = [self.aug(b) for b in tqdm.tqdm(blocks)]
        else:
            self.samples = blocks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sample = self.samples[idx]
        if self.stage == 'train':
            x, y = self.aug(sample)
        else:
            x, y = sample

        xt = torch.tensor(x, dtype=torch.int64)
        yt = torch.tensor(y, dtype=torch.float32)

        return xt, yt

    def __len__(self) -> int:
        return len(self.samples)

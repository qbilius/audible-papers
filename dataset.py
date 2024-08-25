from pathlib import Path
import random
import numpy as np
import torch
import tqdm


class Citations:

    def __init__(self, data_dir: Path, seed: int | None = 0):
        self.rng = random.Random(seed)
        self.lastnames = np.load(data_dir / 'lastnames.npy').tolist()

    def get_year(self) -> int:
        return self.rng.randint(1990, 2024)

    def get_page(self) -> str:
        return f'p. {self.rng.randint(0, 400)}'

    def brackets(self, p=None) -> tuple[str, str]:
        kinds = [('(', ')'), ('[', ']'), ('', '')]
        bo, bc = self.rng.choices(kinds, weights=p, k=1)[0]
        return bo, bc

    def text(self):
        n_authors = self.rng.choices([1, 2, 3, 4], weights=[.1, .1, .1, .7], k=1)[0]
        authors = self.rng.sample(self.lastnames, n_authors)
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

    def numbers(self):
        if self.rng.random() < .9:
            return str(self.rng.randint(1, 110))
        else:
            s, e = self.rng.sample(range(1, 110), 2)
            return f'{s}-{e}'

    def __call__(self):
        n_sources = self.rng.choices([1, 2, 3, 4], weights=[.9, .05, .025, .025], k=1)[0]
        is_number = False
        if self.rng.random() < .5:
            bo, bc = self.brackets(p=[.9, .09, .01])
            citation = '; '.join([self.text() for _ in range(n_sources)])
        else:
            bo, bc = self.brackets()
            citation = ', '.join([self.numbers() for _ in range(n_sources)])
            if self.rng.random() < .5:
                citation = citation.replace(', ', ',')

            # If no parentheses, then this is a pure number
            # This is a superscript citation style ("... as reported previously^1")
            is_number = bo == bc == ''

        return bo + citation + bc, is_number


class Augment:

    def __init__(self,
                 citation_values: list | np.ndarray,
                 cv_isnum: list | np.ndarray,
                 block_size: int,
                 seed: int | None = 0
                 ):
        self.rng = random.Random(seed)
        self.citation_values = list(citation_values)
        self.cv_isnum = list(cv_isnum)
        self.block_size = block_size

    def __call__(self, block: list) -> tuple[list, list]:
        n_citations = self.rng.choices([1, 2], weights=[.95, .05], k=1)[0]

        cits = []
        cits_len = 0
        for _ in range(n_citations):
            cit_idx = self.rng.randint(0, len(self.citation_values) - 1)
            citation_values = self.citation_values[cit_idx]
            citation_values = [c for c in citation_values if c != 50257]
            is_num = self.cv_isnum[cit_idx]
            # Add a space between preceeding text and citation
            # Except superscript style (when is_num is True) that doesn't have a space
            if self.rng.random() > .05 and not is_num:
                citation_values = [220] + citation_values
            else:
                citation_values = citation_values

            cits_len += len(citation_values)

            cits.append([citation_values, is_num])

        pos = max(n_citations, self.block_size - cits_len)
        inds = self.rng.sample(range(pos), n_citations)
        parts = np.split(block, sorted(inds))
        x = parts[0].tolist()
        y = [False] * len(x)
        for part, (cit, is_num) in zip(parts[1:], cits):
            x.extend(cit + part.tolist())
            y += [True] * len(cit) + [False] * len(part)

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

        data = np.array(np.memmap(data_dir / f'{stage}_10.bin', np.uint16, mode='r'))
        blocks = np.array_split(data, len(data) // block_size)
        citation_values = np.load(data_dir / f'{stage}_citations_10.npy')
        cv_isnum = np.load(data_dir / f'{stage}_citations_isnum_10.npy')

        # if self.stage == 'val':
        #     citation_values = citation_values[:1]
        #     cv_isnum = cv_isnum[:1]

        self.augment = Augment(citation_values, cv_isnum, self.block_size)

        if self.stage == 'val':
            self.samples = [self.augment(b) for b in tqdm.tqdm(blocks)]
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
            x, y = self.augment(sample)
        else:  # val is already augmented
            x, y = sample
        xt = torch.tensor(x, dtype=torch.int64)
        yt = torch.tensor(y, dtype=torch.float32)

        return xt, yt

    def __len__(self) -> int:
        return len(self.samples)

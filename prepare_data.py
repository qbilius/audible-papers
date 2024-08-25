from pathlib import Path
import os

import tqdm
import fire
import numpy as np
import tiktoken
import datasets  # Hugging Face

import dataset


with Path('.env').open() as f:
    data_dir = Path(f.read().strip())
enc = tiktoken.get_encoding('gpt2')
n_workers = os.cpu_count() // 2


def trainval_split(max_size_mb: int = None):
    """
    Tokenizes the [gfissore
    /
    arxiv-abstracts-2021](https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021) dataset to a binary file for training.
    """
    dataset = datasets.load_dataset(
        'gfissore/arxiv-abstracts-2021',
        num_proc=n_workers)

    split_dataset = dataset['train'].train_test_split(test_size=1024, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example: dict) -> dict:
        ids = enc.encode_ordinary(example['abstract'].replace('\n', ' '))
        lastname = None if example['submitter'] is None else example['submitter'].split()[-1]
        out = {
            'ids': ids,
            'lastname': lastname,
            'len': len(ids)
        }
        return out

    # tokenize the dataset
    tokenized: dict[str, datasets.Dataset] = split_dataset.map(
        process,
        desc='tokenizing data',
        num_proc=n_workers
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    if max_size_mb is not None:
        n_tokens = max_size_mb * 1024**2 // 2  # convert to bytes and divide by 2, since uint16 is 2 bytes
        suffix = f'_{max_size_mb}'
    else:
        n_tokens = None
        suffix = ''

    lastnames = set()
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        msg = f'{split} has {arr_len} tokens'
        if n_tokens is not None:
            arr_len = min(n_tokens, arr_len)
            msg += f'; {arr_len} will be saved'
        print(msg)

        filename = data_dir / f'{split}{suffix}.bin'
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm.tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])

            # Write into mmap if there is anything left
            end = min(idx + len(arr_batch), n_tokens)
            if len(arr[idx: end]) == 0:
                break
            arr[idx: end] = arr_batch[:end - idx]
            idx += len(arr_batch)
            if idx >= n_tokens:
                break

        arr.flush()

        # accumulate unique last names
        for batch_idx in tqdm.tqdm(range(total_batches), desc=f'writing lastnames.npy'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            names = [n for n in batch['lastname'] if n is not None]
            lastnames.update(set(names))

    np.save(filename.with_name('lastnames.npy'), list(lastnames))
    print(f'There are {len(lastnames)} unique last names')


def citations(block_size=16, max_samples=None, suffix=''):
    c = dataset.Citations(data_dir=data_dir)
    maxval = enc.max_token_value + 1
    for stage in ['train', 'val']:
        data = np.memmap(data_dir / f'{stage}{suffix}.bin', np.uint16, mode='r')

        # generate random citations
        cv = []
        cv_isnum = []
        for _ in tqdm.trange(len(data) // block_size, desc=stage):
            cit, is_number = c()
            cv.append(enc.encode_ordinary(cit))
            cv_isnum.append(is_number)

        # pad each entry
        maxlen = len(max(cv, key=len))
        cv = [cit + [maxval] * (maxlen - len(cit)) for cit in cv]
        np.save(data_dir / f'{stage}_citations{suffix}.npy',
                np.array(cv).astype(np.uint16))
        np.save(data_dir / f'{stage}_citations_isnum{suffix}.npy',
                np.array(cv_isnum).astype(np.bool_))


if __name__ == '__main__':
    fire.Fire()

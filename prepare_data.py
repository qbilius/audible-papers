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


def trainval_split():
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
    lastnames = set()
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f'{split} has {arr_len} tokens')
        filename = data_dir / f'{split}.bin'
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm.tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap if there is anything left
            if len(arr[idx: idx + len(arr_batch)]) == 0:
                break
            arr[idx: idx + len(arr_batch)] = arr_batch

            names = [n for n in batch['lastname'] if n is not None]
            lastnames.update(set(names))

            idx += len(arr_batch)
        arr.flush()

    np.save(filename.with_name('lastnames.npy'), list(lastnames))
    print(f'There are {len(lastnames)} unique last names')


def citations():
    c = dataset.Citations(data_dir=data_dir)
    maxval = enc.max_token_value + 1
    for stage in ['train', 'val']:
        data = np.memmap(data_dir / f'{stage}.bin', np.uint16, mode='r')
        cv = [enc.encode_ordinary(c()) for _ in tqdm.trange(len(data) // 64)]
        # pad each entry
        maxlen = len(max(cv, key=len))
        cv = [cit + [maxval] * (maxlen - len(cit)) for cit in cv]
        np.save(data_dir / f'{stage}_citations.npy',
                np.array(cv).astype(np.uint16))


if __name__ == '__main__':
    fire.Fire()

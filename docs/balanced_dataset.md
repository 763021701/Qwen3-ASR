# Balanced Multilingual Dataset

## Task

Build a balanced multilingual ASR training dataset.

## Rules

1. Treat each dataset as belonging to one language and one source.

2. First perform intra-language balancing:
   - For each language, collect all source datasets under that language.
   - Find the source dataset with the largest number of samples.
   - For every smaller source dataset, repeat its samples until it has at least the same number of samples as the largest source dataset.
   - Randomly sample from each source dataset so that every source dataset in this language has exactly the same number of samples.
   - Concatenate these balanced source datasets to form one language pool.

3. Then perform inter-language balancing:
   - Compare all language pools.
   - Find the largest language pool.
   - For every smaller language pool, repeat its samples until it has at least the same number of samples as the largest language pool.
   - Randomly sample from each language pool so that every language has exactly the same number of samples.

4. Concatenate all final language pools.

5. Shuffle the final dataset.

6. Use a fixed random seed for all random operations.

## Result

The final dataset contains the same number of samples for every language. Within each language, every source dataset contributes the same number of samples.

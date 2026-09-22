import csv
import random
from collections import defaultdict

from torch.utils.data import Sampler, Subset


class CategoryAwareBatchSampler(Sampler):
    """
    Batch sampler for THINGS-EEG2 training.

    The sampler changes only which samples are placed in the same mini-batch.
    It does not change the loss function or duplicate samples.

    For batch_size=64, categories_per_batch=4, and
    samples_per_category=12, a category-aware batch is built as:

        12 samples from category A
        12 samples from category B
        12 samples from category C
        12 samples from category D
        16 filler samples
        -----------------
        64 samples

    A-D are distinct categories within each category-aware batch.

    THINGSplus categories are multi-label. At the beginning of each epoch,
    each categorized class is assigned to exactly one of its available
    categories using a deterministic RNG (seed + epoch). This prevents the
    same sample from appearing in multiple category pools.

    Samples that cannot be placed in a complete category-aware batch are not
    discarded. They are shuffled and emitted as ordinary random batches.
    Thus all possible full batches are retained without oversampling.
    """

    def __init__(
        self,
        subset,
        category_tsv,
        batch_size=64,
        categories_per_batch=4,
        samples_per_category=12,
        seed=42,
    ):
        if not isinstance(subset, Subset):
            raise TypeError(
                "CategoryAwareBatchSampler expects torch.utils.data.Subset."
            )

        self.subset = subset
        self.category_tsv = category_tsv
        self.batch_size = batch_size
        self.categories_per_batch = categories_per_batch
        self.samples_per_category = samples_per_category
        self.seed = seed
        self.epoch = 0

        grouped = categories_per_batch * samples_per_category
        self.filler_per_batch = batch_size - grouped

        if self.filler_per_batch < 0:
            raise ValueError(
                "categories_per_batch * samples_per_category "
                "must not exceed batch_size."
            )

        base_dataset = subset.dataset

        if not hasattr(base_dataset, "labels"):
            raise AttributeError("Base dataset must provide labels.")

        if not hasattr(base_dataset, "text"):
            raise AttributeError("Base dataset must provide text.")

        self.class_to_categories = self._load_categories(
            base_dataset.text,
            category_tsv,
        )

        self.positions_by_class = defaultdict(list)

        # DataLoader with batch_sampler expects indices relative to the Subset.
        for subset_position, full_index in enumerate(subset.indices):
            class_id = int(base_dataset.labels[full_index])
            self.positions_by_class[class_id].append(subset_position)

        self.num_categorized_classes = sum(
            1
            for class_id in self.positions_by_class
            if self.class_to_categories.get(class_id)
        )

    @staticmethod
    def _load_categories(class_texts, category_tsv):
        concept_to_categories = defaultdict(set)

        with open(category_tsv, encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                concept_to_categories[row["uniqueID"]].add(
                    row["category"]
                )

        class_to_categories = {}
        prefix = "This picture is "

        for class_id, text in enumerate(class_texts):
            if text.startswith(prefix):
                class_name = text[len(prefix):]
            else:
                class_name = text

            class_to_categories[class_id] = sorted(
                concept_to_categories.get(class_name, set())
            )

        return class_to_categories

    def __len__(self):
        # Same number of full batches as DataLoader(..., drop_last=True).
        return len(self.subset) // self.batch_size

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1

        rng = random.Random(self.seed + epoch)

        category_pools = defaultdict(list)
        filler_pool = []

        # Assign each class to exactly one category for this epoch.
        # Uncategorized classes go directly to the filler pool.
        for class_id, positions in self.positions_by_class.items():
            positions = list(positions)
            rng.shuffle(positions)

            categories = self.class_to_categories.get(class_id, [])

            if categories:
                chosen_category = rng.choice(categories)
                category_pools[chosen_category].extend(positions)
            else:
                filler_pool.extend(positions)

        # Split each category into fixed-size same-category blocks.
        # Remainders are preserved as filler.
        blocks_by_category = defaultdict(list)

        for category, pool in category_pools.items():
            rng.shuffle(pool)

            num_blocks = len(pool) // self.samples_per_category

            for block_index in range(num_blocks):
                start = block_index * self.samples_per_category
                end = start + self.samples_per_category
                blocks_by_category[category].append(pool[start:end])

            remainder_start = num_blocks * self.samples_per_category
            filler_pool.extend(pool[remainder_start:])

        for blocks in blocks_by_category.values():
            rng.shuffle(blocks)

        rng.shuffle(filler_pool)

        total_batches = len(self)
        category_aware_batches = []
        filler_cursor = 0

        # Build batches from distinct categories while possible.
        while len(category_aware_batches) < total_batches:
            active_categories = [
                category
                for category, blocks in blocks_by_category.items()
                if blocks
            ]

            if len(active_categories) < self.categories_per_batch:
                break

            if (
                self.filler_per_batch > 0
                and filler_cursor + self.filler_per_batch > len(filler_pool)
            ):
                break

            chosen_categories = rng.sample(
                active_categories,
                self.categories_per_batch,
            )

            batch = []

            for category in chosen_categories:
                batch.extend(blocks_by_category[category].pop())

            if self.filler_per_batch > 0:
                batch.extend(
                    filler_pool[
                        filler_cursor:
                        filler_cursor + self.filler_per_batch
                    ]
                )
                filler_cursor += self.filler_per_batch

            rng.shuffle(batch)
            category_aware_batches.append(batch)

        print(
            "[CategoryBatchSampler] "
            f"epoch={epoch + 1} "
            f"category-aware batches="
            f"{len(category_aware_batches)}/{total_batches} "
            f"categorized classes="
            f"{self.num_categorized_classes}/"
            f"{len(self.positions_by_class)}"
        )

        for batch in category_aware_batches:
            yield batch

        # Preserve all unused samples and use them in ordinary random batches.
        remaining = list(filler_pool[filler_cursor:])

        for blocks in blocks_by_category.values():
            for block in blocks:
                remaining.extend(block)

        rng.shuffle(remaining)

        remaining_batches = total_batches - len(category_aware_batches)

        for batch_index in range(remaining_batches):
            start = batch_index * self.batch_size
            end = start + self.batch_size
            batch = remaining[start:end]

            if len(batch) == self.batch_size:
                yield batch

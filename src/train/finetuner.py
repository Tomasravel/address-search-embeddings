#%%
import pandas as pd
import os
import yaml
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, InputExample, losses
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset
from logs.logger import Logger
from train.preprocessing.finetuning_dataset import FinetuningDataset 
from train.constants import ColumnCategories as ColCat
from datetime import datetime
#%%

class FineTuner:
    """
    Generic class for fine-tuning Sentence-Transformers with
    * MultipleNegativesRankingLoss
    * TripletLoss

    The dataset must contain:
        • a column named **anchor**
        • ≥1 columns whose name includes the word **positive**
        • (optional) ≥1 columns whose name includes **negative**
    """

    # --------------- INIT & CONFIG -----------------
    def __init__(self, settings):
        # config
        self.settings = settings
        # Save current date and time for output directory naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_dir = os.path.join(self.settings["OUTPUT_DIR"], f"{timestamp}_{self.settings['NAME']}")
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, ".debug"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "copies"), exist_ok=True)
        with open(os.path.join(self.out_dir, ".debug", "train_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.settings, f, ensure_ascii=False, indent=4)

        self.logger = Logger(name=__class__.__name__, settings=settings).get()

        self.model = SentenceTransformer(self.settings["BASE_MODEL_PATH"])

        self.data = FinetuningDataset(settings)

    def _generate_anchor_positive_pairs(self):
        """
        Generate anchor-positive pairs from the DataFrame.
        """
        if not self.data.columns[ColCat.POSITIVE]:
            raise ValueError("To generate pairs a *positive* column is required")
        anchor_positive_pairs = []
        for _, row in self.data.df.iterrows():
            anchor = row[self.data.ANCHOR_NAME]
            for pc in self.data.columns[ColCat.POSITIVE] + self.data.columns[ColCat.ANCHOR]:
                if pc == self.data.ANCHOR_NAME:
                    continue
                pos = row[pc]
                if isinstance(anchor, str) and isinstance(pos, str) and pos != "":
                    anchor_positive_pairs.append(
                        InputExample(texts=[anchor, pos])
                    )
        self.anchor_positive_pairs = anchor_positive_pairs
        pairs_path = os.path.join(self.out_dir, ".debug", "anchor_positive_pairs.csv")
        pd.DataFrame(
            [ex.texts for ex in anchor_positive_pairs],
            columns=["anchor", "positive"]
        ).to_csv(pairs_path, index=False)
        self.logger.info("Prepared %d pairs", len(anchor_positive_pairs))
    
    def _generate_triplets(self):
        """
        Generate triplets (anchor, positive, negative) from the DataFrame.
        """
        if not self.data.columns[ColCat.NEGATIVE]:
            raise ValueError("To generate triplets a *negative* column is required")
        triplets = []
        for _, row in self.data.df.iterrows():
            anchor = row[self.data.ANCHOR_NAME]
            for pc in self.data.columns[ColCat.POSITIVE] + self.data.columns[ColCat.ANCHOR]:
                if pc == self.data.ANCHOR_NAME:
                    continue
                pos = row[pc]
                for nc in self.data.columns[ColCat.NEGATIVE]:
                    neg = row[nc]
                    if isinstance(anchor, str) and isinstance(pos, str) and isinstance(neg, str):
                        triplets.append(InputExample(texts=[anchor, pos, neg]))
        self.triplets = triplets
        self.logger.info("Prepared %d triplets", len(triplets))

    def _multiple_negative_ranking_loss_fit(self) -> None:
        """Fine-tuning con MultipleNegativesRankingLoss (solo anchor-positive)."""
        # Group by anchor
        anchors = []
        positives = []
        for ex in self.anchor_positive_pairs:
            anchors.append(ex.texts[0])
            positives.append(ex.texts[1])

        dataset = Dataset.from_dict({
            "anchor": anchors,
            "positive": positives,
        })
        # Create a custom dataset without duplicated anchors per batch
        # (this is important for MNRL)
        loss_fn = losses.MultipleNegativesRankingLoss(self.model)
        training_args = SentenceTransformerTrainingArguments(
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            output_dir=os.path.join(self.out_dir, "checkpoints"),
            save_total_limit=self.settings["CHECKPOINT_SAVE_TOTAL_LIMIT"], 
            save_steps=self.settings["CHECKPOINT_SAVE_STEPS"],
            per_device_train_batch_size=self.settings["BATCH_SIZE"],
            num_train_epochs=self.settings["EPOCHS"],
            warmup_steps=int(len(dataset) * self.settings["EPOCHS"] * 0.1),
            logging_dir=os.path.join(self.out_dir, "logs"),
            disable_tqdm=not self.settings["SHOW_PROGRESS_BAR"],
        )
        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            loss=loss_fn,
        )
        self.logger.info("Starting training with SentenceTransformerTrainer + NO_DUPLICATES batching …")
        trainer.train()

    def _triplet_loss_fit(self) -> None:
        """Fine-tuning con TripletLoss (anchor, positive, negative)."""
        loader = DataLoader(self.triplets, shuffle=True, batch_size=self.settings["BATCH_SIZE"])
        loss_fn = losses.TripletLoss(self.model)

        self.logger.info("Starting training (TripletLoss) …")
        self.model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=self.settings["EPOCHS"],
            warmup_steps=int(len(loader) * self.settings["EPOCHS"] * 0.1),
            show_progress_bar=self.settings["SHOW_PROGRESS_BAR"],
            checkpoint_path=os.path.join(self.out_dir, "checkpoints"),
            checkpoint_save_steps=self.settings["CHECKPOINT_SAVE_STEPS"],  
            checkpoint_save_total_limit=self.settings["CHECKPOINT_SAVE_TOTAL_LIMIT"],
        )
    
    def process_dataset(self) -> None:
        self.data.load_dataset()
        if self.settings["AUGMENTATION"]:
            self.data.augment_dataset()
        self.data.df.to_csv(
            os.path.join(self.out_dir, ".debug", "processed_dataset.csv"),
            index=False
        )

    def train(self) -> None:
        """
        loss ∈ {'mnr', 'triplet'}
        """
        self.process_dataset()
        if self.settings["LOSS"] == "mnr":
            self._generate_anchor_positive_pairs()
            self._multiple_negative_ranking_loss_fit()
        elif self.settings["LOSS"] == "triplet":
            self._generate_triplets()
            self._triplet_loss_fit()
        else:
            raise ValueError(f"Loss '{self.settings['LOSS']}' no soportada.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 src/train/finetuner.py path/al/train_config.yaml")
        sys.exit(1)

    train_config_path = sys.argv[1]
    with open(train_config_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    finetuner = FineTuner(settings)
    finetuner.train()
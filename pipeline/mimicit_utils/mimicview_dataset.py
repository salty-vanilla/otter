import torch

from .mimicit_dataset import MimicitDataset


class MimicViewDataset(MimicitDataset):
    def process_image_text_pair(self, index):
        cur_train_id = self.train_data_list[index]
        cur_train_id_for_ins = cur_train_id.split("=")[0]
        (
            instruction_id,
            instruction,
            answer,
            image_ids,
            in_context_example_ids,
        ) = (
            cur_train_id_for_ins,
            self.dataset[cur_train_id_for_ins]["instruction"],
            self.dataset[cur_train_id_for_ins]["answer"],
            self.dataset[cur_train_id_for_ins]["image_ids"],
            self.train_config[cur_train_id],
        )
        inst_format = self.inst_format

        patch_images, all_texts = self.process_llava(
            instruction_id,
            instruction,
            answer,
            image_ids,
            in_context_example_ids,
            inst_format=inst_format,
        )

        all_text = self.tokenizer(
            f"{all_texts}",
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,  # for current 2k mpt/llama model, setting to 2048 causes error (2042 works)
        )

        all_item = all_text["input_ids"].squeeze(0)
        all_item_mask = all_text["attention_mask"].squeeze(0)

        all_item = torch.cat([self.bos_item, all_item, self.eos_item])
        all_item_mask = torch.cat([self.bos_mask, all_item_mask, self.eos_mask])

        example = {
            "id": instruction_id,
            "source": all_item,
            "text_mask": all_item_mask,
            "patch_images": patch_images,
        }

        return example

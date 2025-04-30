"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.tracking import Tracking

from verl import DataProto

WorkerType = Type[Worker]


def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(tensors=item.batch,  # TensorDict is already in correct format
                               non_tensors=item.non_tensor_batch,  # Dict is already in correct format
                               meta_info=item.meta_info)


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(response_mask=response_mask, prompt_length=prompt_length, response_length=response_length, )


def compute_data_metrics(batch, ):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)
    max_response_length = batch.batch['responses'].shape[-1]
    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()

    max_prompt_length = prompt_mask.size(-1)
    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    metrics = {  # score
        'critic/score/mean': torch.mean(sequence_score).detach().item(), 'critic/score/max': torch.max(sequence_score).detach().item(),
        'critic/score/min': torch.min(sequence_score).detach().item(),  # reward
        'critic/rewards/mean': torch.mean(sequence_reward).detach().item(), 'critic/rewards/max': torch.max(sequence_reward).detach().item(),
        'critic/rewards/min': torch.min(sequence_reward).detach().item(),  # response length
        'response_length/mean': torch.mean(response_length).detach().item(), 'response_length/max': torch.max(response_length).detach().item(),
        'response_length/min': torch.min(response_length).detach().item(),
        'response_length/clip_ratio': torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),  # prompt length
        'prompt_length/mean': torch.mean(prompt_length).detach().item(), 'prompt_length/max': torch.max(prompt_length).detach().item(),
        'prompt_length/min': torch.min(prompt_length).detach().item(), 'prompt_length/clip_ratio': torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(), }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {'gen': num_response_tokens, **{name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_sft', ]}, }

    return {**{f'timing_s/{name}': value for name, value in timing_raw.items()},
            **{f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())}, }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RaySFTTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self, config, tokenizer, role_worker_mapping: dict[Role, WorkerType], resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup, reward_fn=None, val_reward_fn=None):
        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        self._create_dataloader()

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files, tokenizer=self.tokenizer, prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length, filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        train_batch_size = self.config.data.train_batch_size
        if self.config.trainer.rejection_sample:  # set to True
            train_batch_size *= self.config.trainer.rejection_sample_multiplier  # ~50% are rejected samples, so compensated by 2 times
            train_batch_size = int(train_batch_size)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files, tokenizer=self.tokenizer, prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length, filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset, batch_size=len(self.val_dataset), shuffle=True, drop_last=True, collate_fn=collate_fn)
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id, 'recompute_log_prob': False, 'do_sample': False,
                                        'validate': True, }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = self.config.actor_rollout_ref.rollout.val_temperature
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            test_batch = test_batch.union(test_output_gen_batch)
            print('Validation: Generation end.')

            # evaluate using reward_function for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.actor_rollout_ref, role='actor_rollout',
                                                     reward_config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor', f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        """
        logger = Tracking(project_name=self.config.trainer.project_name, experiment_name=self.config.trainer.experiment_name, default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False): return

        # we start from step 1
        self.global_steps += 1
        for _ in range(self.config.trainer.total_epochs):

            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                metrics = {}
                timing_raw = {}

                # get seq ids
                batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        batch = self.actor_rollout_wg.generate_sequences(batch)

                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if not self.config.actor_rollout_ref.rollout.compute_reward:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch['token_level_scores'] = reward_tensor
                        else:
                            reward_tensor = batch.batch['token_level_scores']
                        # Rejection sampling based on group rewards
                        uids = batch.non_tensor_batch['uid']
                        # print('uids')  # todo
                        # print(uids.shape)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        if self.config.actor_rollout_ref.rollout.n > 1:
                            solve_none = 0
                            solve_all = 0
                            unique_uids = np.unique(uids)
                            # print('unique:', unique_uids.shape)
                            for uid in unique_uids:
                                uid_mask = torch.from_numpy((uids == uid))   # todo: from_numpy?
                                uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                                # Check if all rewards are 0 or all are 1 for this uid
                                if (uid_rewards == 0).all():   # todo: trigger self-taught
                                    valid_mask[uid_mask] = False
                                    solve_none += 1
                                elif (uid_rewards == 1).all() and self.config.trainer.filter_too_easy:   # todo: remove too easy? but only for group sampling
                                    valid_mask[uid_mask] = False
                                    solve_all += 1

                            # Log to metrics
                            metrics['batch/solve_none'] = solve_none
                            metrics['batch/solve_all'] = solve_all
                            metrics['batch/solve_partial'] = len(unique_uids) - solve_none - solve_all

                        per_sequence_reward = reward_tensor.sum(dim=-1)  # [B]   # todo: adjustive difficulty weight
                        keep = per_sequence_reward > 0  # only retain any sequence that has at least one correct token
                        valid_mask = valid_mask & keep

                        # If no valid samples remain, skip this batch and get a new one
                        if not valid_mask.any(): continue

                        # Filter batch to keep only valid samples
                        batch = batch[valid_mask]
                        batch = dataprotoitem_to_dataproto(batch)
                        # Round down to the nearest multiple of world size
                        num_trainer_replicas = self.actor_rollout_wg.world_size
                        max_batch_size = (batch.batch['input_ids'].shape[0] // num_trainer_replicas) * num_trainer_replicas
                        if not max_batch_size: continue  # give up, you got everything either all wrong or right.

                        size_mask = torch.zeros(batch.batch['input_ids'].shape[0], dtype=torch.bool)
                        size_mask[:max_batch_size] = True
                        batch = batch[size_mask]
                        batch = dataprotoitem_to_dataproto(batch)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # supervised fine-tuning update
                    with _timer('update_sft', timing_raw):
                        sft_output = self.actor_rollout_wg.update_actor(batch, sft=True)
                    sft_output_metrics = reduce_metrics(sft_output.meta_info['metrics'])
                    metrics.update(sft_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, ))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1
                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

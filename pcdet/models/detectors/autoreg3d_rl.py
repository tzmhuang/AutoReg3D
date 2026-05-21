import torch
import copy

from .autoreg3d import AutoReg3D
from ...utils.rl_utils import reward_fn, nanmin, nanmax
from ...utils.commu_utils import concat_all_gather, all_reduce, get_world_size


class AutoReg3DRL(AutoReg3D):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.training_mode = 'rl'

        # RL training parameters
        self.rl_config = model_cfg.get('RL_CONFIG', {})
        self.importance_sampling_level = self.rl_config.get('IMPORTANCE_SAMPLING_LEVEL', 'token')
        self.epsilon_low = self.rl_config.get('EPSILON', 0.2)
        self.epsilon_high = self.rl_config.get('EPSILON_HIGH', self.epsilon_low)
        self.beta = self.rl_config.get('BETA', 0.0)  # KL penalty coefficient
        self.group_size = self.rl_config.get('NUM_GENERATION_IN_GROUP', 1)  # num samples each group
        self.temperature = self.rl_config.get('SAMPLING_TEMPERATURE', 1.0)
        self.ignore_velocity_in_rl_loss = self.rl_config.get('IGNORE_VELOCITY_IN_RL_LOSS', False)
        self.use_greedy_baseline = self.rl_config.get('USE_GREEDY_BASELINE', False)
        self.entropy_coeff = self.rl_config.get('ENTROPY_COEFF', 0.0)

        # KL penalty/logging configuration (backward-compatible defaults)
        # - Default keeps behavior unchanged: no extra compute/logs and no penalty.
        self.kl_log = self.rl_config.get('LOG_KL', False)
        self.kl_penalty_level = self.rl_config.get('KL_PENALTY_LEVEL', 'sequence')  # 'sequence' or 'token'
        self.kl_temperature = self.rl_config.get('KL_TEMPERATURE', 1.0)
        self.kl_ref_scope = self.rl_config.get('KL_REF_SCOPE', 'head')  # 'head' only; 'full' not implemented
        # Choose reference policy source. If not specified, default to 'clone' only when needed.
        self.kl_ref_mode = self.rl_config.get('KL_REF_MODE', None)
        if self.kl_ref_mode is None:
            self.kl_ref_mode = 'clone' if (self.beta > 0 or self.kl_log) else 'none'

        self._ref_dense_head = []
        self.world_size = get_world_size()

        # Decode mode is applied per-call in forward(): TRAINING_DECODE_MODE for RL
        # rollouts (exploration), and the head's configured DECODE_MODE for eval.
        self._train_decode_mode = self.rl_config.get('TRAINING_DECODE_MODE', 'sample')
        self._eval_decode_mode = self.dense_head.decode_mode

    def init_reference_model(self):
        """Build a frozen reference head for the KL penalty / KL logging.

        Called by the training script only when a reference policy is needed
        (beta > 0 or kl_log, with kl_ref_mode != 'none'). Only head-only KL is
        supported; KL_REF_SCOPE='full' is not implemented.
        """
        if (self.beta > 0 or self.kl_log) and self.kl_ref_mode != 'none':
            if self.kl_ref_scope == 'full':
                raise NotImplementedError(
                    "Full-model KL reference (KL_REF_SCOPE='full') is not implemented yet. "
                    "Only head-only KL is currently supported."
                )
            # Deep-copy the current head as the frozen reference
            # (common case: SFT init already loaded).
            ref = copy.deepcopy(self.dense_head)
            for p in ref.parameters():
                p.requires_grad = False
            ref.eval()
            self._ref_dense_head = [ref]

    @property
    def ref_dense_head(self):
        return self._ref_dense_head[0] if self._ref_dense_head else None

    def forward(self, batch_dict):
        """Override to add RL training path; delegates to super for inference."""
        if self.training:
            self.dense_head.decode_mode = self._train_decode_mode
            batch_dict = self.forward_features(batch_dict)

            loss, tb_dict = self.forward_rl_train(batch_dict)
            ret_dict = {
                'loss': loss
            }
            disp_dict = {}
            return ret_dict, tb_dict, disp_dict
        else:
            self.dense_head.decode_mode = self._eval_decode_mode
            return super().forward(batch_dict)

    def forward_rl_train(self, batch_dict):
        """Forward function for RL training.

        Returns:
            loss: Tensor scalar representing the RL loss
            tb_dict: dict for tensorboard logging
        """
        log_dict = {}

        # Greedy baseline for advantage calculation
        greedy_rewards = None
        if self.use_greedy_baseline:
            greedy_decoded = self.greedy_rollout(batch_dict)
            greedy_rewards, _ = self.compute_reward(batch_dict, greedy_decoded, {}, group_size=1)  # (B,)

        # Generate sequences and compute log-probabilities and entropies
        batch_generation, decoded_preds, batch_dict_exp, log_dict = self.rl_rollout(batch_dict, log_dict)

        # Compute rewards
        rewards, log_dict = self.compute_reward(batch_dict, decoded_preds, log_dict)  # (B*G,)

        # Compute advantages (+ logs reward/advantage stats)
        advantages, log_dict = self._compute_advantages(rewards, log_dict, greedy_rewards=greedy_rewards)

        # Compute RL loss
        rl_loss, log_dict = self.compute_rl_loss(
            batch_dict_exp, batch_generation, advantages, log_dict
        )

        return rl_loss, log_dict


    def _repeat_for_groups(self, batch_dict, repeat: int):
        if repeat == 1:
            return batch_dict
        new = {}
        bs = batch_dict['batch_size']
        for k, v in batch_dict.items():
            if torch.is_tensor(v) and v.size(0) == bs:
                new[k] = v.repeat_interleave(repeat, dim=0) # (B*repeat, ...)
            else:
                new[k] = v
        return new


    def rl_rollout(self, batch_dict, log_dict, group_size=None):
        """Generate sequences and compute log-probabilities and entropies for RL training.

        Args:
            group_size: override for self.group_size (number of samples per scene).

        Returns:
            batch_generation: dict containing:
                'pred_seqs': Tensor (B*G, L) predicted token sequences including BOS
                'old_per_tok_logps': Tensor (B*G, L-1) log-probabilities of generated tokens (no BOS)
                'valid_mask': Tensor (B*G, L-1) mask indicating valid (non-PAD) tokens for logps/entropies
            decoded_preds: list of dicts, each containing:
                'pred_boxes': Tensor (N_boxes, 7) predicted boxes
                'pred_labels': Tensor (N_boxes,) predicted class labels
        """
        # Expand batch by group size G
        G = int(group_size if group_size is not None else self.group_size)
        batch_dict_exp = self._repeat_for_groups(batch_dict, G)

        # Generate on expanded batch (B*G, ...)
        batch_generation = self.dense_head.generate_with_logps_and_entropies(
            batch_dict_exp, tokenizer=self.tokenizer, temperature=self.temperature, compute_entropy=False
        )
        pred_seqs = batch_generation['pred_seqs']       # (B*G, L)
        valid_mask = batch_generation['valid_mask']     # (B*G, L-1)


        # decode generation for reward computation
        # NOTE: No filtering of noise boxes here
        decoded_preds = self.simple_decode(pred_seqs)   # list of B*G dicts

        # ---------------------------------------------------
        # logging
        # ---------------------------------------------------

        completion_tok_lengths = valid_mask.sum(-1) + 1  # (B,)
        completion_tok_lengths_gather = concat_all_gather(completion_tok_lengths) if self.world_size > 1 else completion_tok_lengths
        log_dict.update({
            'rl/completion/mean_tok_length': completion_tok_lengths_gather.float().mean().item(),
            'rl/completion/max_tok_length': completion_tok_lengths_gather.max().item(),
            'rl/completion/min_tok_length': completion_tok_lengths_gather.min().item(),
        })

        # valid boxes ratio
        box_num = torch.tensor([p['pred_labels'].numel() for p in decoded_preds], device=pred_seqs.device)  # shape (B,)
        # Ensure Python scalars to avoid constructing tensors from tensors of different devices/dtypes
        nonnoise_box_nums = torch.tensor([(p['pred_labels'] > 0).sum().item() for p in decoded_preds], device=pred_seqs.device)  # shape (B,)
        nonnoise_box_ratio = nonnoise_box_nums.sum() / box_num.sum().clamp(min=1.0) # scalar

        log_dict.update({
            'rl/completion/box_num': all_reduce(box_num.float().mean(), op='sum', average=True).item(),
            'rl/completion/nonnoise_box_num': all_reduce(nonnoise_box_nums.float().mean(), op='sum', average=True).item(),
            'rl/completion/nonnoise_box_ratio': all_reduce(nonnoise_box_ratio, op='sum', average=True).item(),
        })

        return batch_generation, decoded_preds, batch_dict_exp, log_dict

    @torch.no_grad()
    def greedy_rollout(self, batch_dict):
        """Generate B greedy sequences as baseline for advantage calculation. No gradient needed."""
        original_mode = self.dense_head.decode_mode
        self.dense_head.decode_mode = 'greedy'
        was_training = self.dense_head.training
        self.dense_head.eval()

        greedy_preds, _ = self.dense_head.generate(
            batch_dict, tokenizer=self.tokenizer, temperature=1.0
        )  # (B, L)

        if was_training:
            self.dense_head.train()
        self.dense_head.decode_mode = original_mode

        return self.simple_decode(greedy_preds)  # list of B dicts

    def compute_reward(self, batch_dict, decoded_preds, log_dict, group_size=None):
        """Compute rewards for RL training based on decoded predictions and ground-truth.

        Args:
            group_size: override group size (default: self.group_size). Use group_size=1 for greedy baseline.

        Returns:
            rewards: Tensor (B*G,) reward values for each sample in the batch
        """
        B = batch_dict['batch_size']
        G = group_size if group_size is not None else int(getattr(self, 'group_size', 1))

        rewards = torch.zeros((B*G,), device=batch_dict['gt_boxes'].device)
        recall_rewards = torch.zeros((B*G,), device=batch_dict['gt_boxes'].device)
        precision_rewards = torch.zeros((B*G,), device=batch_dict['gt_boxes'].device)

        # compute IoU-based rewards or other metrics here
        gt_boxes = batch_dict['gt_boxes'][:, :, :7]  # list of (N_gt, 7) per batch element
        gt_labels = batch_dict['gt_boxes'][:, :, -1]  # list of (N_gt,) per batch element


        for idx in range(B*G):
            bi = idx//G  # batch index
            pred_boxes = decoded_preds[idx]['pred_boxes']
            pred_labels = decoded_preds[idx]['pred_labels']

            gt_boxes_batch = gt_boxes[bi]
            gt_labels_batch = gt_labels[bi]

            # filter out PAD and Noise boxes from gt and predictions for reward computation
            gt_boxes_batch = gt_boxes_batch[gt_labels_batch > 0]
            gt_labels_batch = gt_labels_batch[gt_labels_batch > 0]
            pred_boxes = pred_boxes[pred_labels > 0]
            pred_labels = pred_labels[pred_labels > 0]

            if gt_boxes_batch.shape[0] == 0 or pred_boxes.shape[0] == 0:
                # No ground-truth boxes, assign zero reward
                reward = 0.0
                recall_reward = 0.0
                precision_reward = 0.0
                if gt_boxes_batch.shape[0] == 0 and pred_boxes.shape[0] == 0:
                    # both empty, perfect score
                    reward = 1.0
            else:
                reward, recall_reward, precision_reward = reward_fn(
                    pred_boxes, pred_labels,
                    gt_boxes_batch, gt_labels_batch,
                ) 

            rewards[idx] = reward
            recall_rewards[idx] = recall_reward
            precision_rewards[idx] = precision_reward

        # Gather and log precision/recall rewards
        recall_reward_gather = concat_all_gather(recall_rewards) if self.world_size > 1 else recall_rewards
        precision_reward_gather = concat_all_gather(precision_rewards) if self.world_size > 1 else precision_rewards
        log_dict.update({
            'rl/reward/mean_recall_reward': recall_reward_gather.float().mean().item(),
            'rl/reward/mean_precision_reward': precision_reward_gather.float().mean().item(),
        })

        return rewards, log_dict

    def _compute_advantages(self, rewards, log_dict, greedy_rewards=None, group_size=None):
        """Compute normalized advantages from rewards using GRPO normalization, and log reward/advantage stats.

        Args:
            rewards: (B*G,) flat reward tensor
            log_dict: dict for tensorboard logging (updated in-place)
            greedy_rewards: (B,) optional greedy baseline rewards
            group_size: override for self.group_size (e.g. N*G for rejection sampling)

        Returns:
            advantages: (B*G,) normalized advantages
            log_dict: updated logging dict
        """
        G = int(group_size if group_size is not None else self.group_size)
        BG = rewards.size(0)
        assert BG % G == 0, "Reward count must be multiple of group size"

        rewards_grouped = rewards.view(-1, G)  # (B, G)

        mean_grouped_rewards = rewards_grouped.mean(1, keepdim=True)  # (B, 1)
        max_grouped_rewards, _ = rewards_grouped.max(1, keepdim=True)  # (B, 1)
        min_grouped_rewards, _ = rewards_grouped.min(1, keepdim=True)  # (B, 1)

        if greedy_rewards is not None:
            baseline = greedy_rewards.unsqueeze(1)  # (B, 1)
        else:
            baseline = mean_grouped_rewards  # (B, 1)

        advantages = rewards_grouped - baseline  # (B, G)
        std_rewards = rewards_grouped.std(1, keepdim=True)  # (B, 1)
        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # (B, 1)
        advantages = advantages / (std_rewards + 1e-4)  # (B, G)
        advantages = advantages.view(-1)  # (B*G,)

        # ---------------------------------------------------
        # logging: reward and advantage statistics
        # ---------------------------------------------------
        log_dict.update({
            'rl/reward_mean': all_reduce(mean_grouped_rewards.mean().detach(), op='sum', average=True).item(),
            'rl/reward_max': all_reduce(max_grouped_rewards.mean().detach(), op='max').item(),
            'rl/reward_min': all_reduce(min_grouped_rewards.mean().detach(), op='min').item(),
            'rl/reward_std': all_reduce(std_rewards.mean().detach(), op='sum', average=True).item(),
            'rl/advantages_mean': all_reduce(advantages.mean().detach(), op='sum', average=True).item(),
            'rl/frac_reward_zero_std': all_reduce(is_std_zero.float().mean().detach(), op='sum', average=True).item(),
        })

        # greedy-specific logs
        if greedy_rewards is not None:
            beats_greedy = (rewards_grouped > greedy_rewards.unsqueeze(1)).float().mean()
            log_dict.update({
                'rl/reward/greedy_mean': all_reduce(greedy_rewards.mean().detach(), op='sum', average=True).item(),
                'rl/reward/samples_beat_greedy': all_reduce(beats_greedy.detach(), op='sum', average=True).item(),
            })

        return advantages, log_dict

    def compute_rl_loss(self, batch_dict_exp, batch_generation, advantages, log_dict):
        """Compute RL loss using GRPO.

        Args:
            batch_dict_exp: expanded data batch dictionary
            batch_generation: dict containing:
                'pred_seqs': Tensor (B*G, L) predicted token sequences including BOS
                'old_per_tok_logps': Tensor (B*G, L-1) log-probabilities of generated tokens (no BOS)
                'valid_mask': Tensor (B*G, L-1) mask indicating valid (non-PAD) tokens for logps/entropies
            advantages: Tensor (B*G,) pre-computed normalized advantages
            log_dict: dict for tensorboard logging (updated in-place)

        Returns:
            loss: Tensor scalar representing the RL loss
            log_dict: updated logging dict
        """

        pred_seqs = batch_generation['pred_seqs']                         # shape (B*G, L)
        valid_mask = batch_generation['valid_mask']                       # shape (B*G, L-1)
        old_per_tok_logps = batch_generation['old_per_tok_logps']         # shape (B*G, L-1)

        # ---------------------------------------------------
        # mask out velocity tokens if needed
        # --------------------------------------------------
        if self.tokenizer.predict_velocity and self.ignore_velocity_in_rl_loss:
            step_pos = torch.arange(valid_mask.size(1), device=valid_mask.device) % self.tokenizer.tok_per_seq  # (L-1,)
            # vx/vy are present only when predict_velocity=True and SequenceExpandedVocabTokenizer sets box_cols accordingly
            vxvy_rel = [self.tokenizer.box_cols[-1], self.tokenizer.box_cols[-2]]
            is_vel = (step_pos == vxvy_rel[0]) | (step_pos == vxvy_rel[1])
            valid_mask = valid_mask & (~is_vel.unsqueeze(0))    # (B*G, L-1)


        # ---------------------------------------------------
        # get online log-probabilities of the generated sequences
        # --------------------------------------------------

        was_training = self.dense_head.training
        self.dense_head.eval()  # set to eval mode to avoid dropout randomness
        per_tok_logps, per_tok_entropies, _ = self.dense_head._get_token_logps_and_entropies(
            batch_dict_exp, pred_seqs, self.tokenizer,
            temperature=self.temperature, compute_entropy=True
        )  # (B*G, L-1), (B*G, L-1), None
        if was_training:
            self.dense_head.train()  # back to train mode

        # ---------------------------------------------------
        # compute KL(new || ref) for logging/penalty (optional)
        # --------------------------------------------------
        kl_token_mean = None
        kl_seq = None
        if self.ref_dense_head is not None:
            # ---------------------------------------------------
            # Recompute log-probs at KL temperature (typically 1.0) to decouple from sampling temperature
            # If temperatures match, reuse computed log-probs; otherwise, recompute with grad at KL temperature
            if abs(self.temperature - self.kl_temperature) > 1e-6:
                was_training_head = self.dense_head.training
                self.dense_head.eval()
                new_logps_kl, _, valid_mask_kl = self.dense_head._get_token_logps_and_entropies(
                    batch_dict_exp, pred_seqs, self.tokenizer,
                    temperature=self.kl_temperature, compute_entropy=False
                )  # (B*G, L-1)
                if was_training_head:
                    self.dense_head.train()
            else:
                # Reuse previously computed log-probs assuming temperature matches
                new_logps_kl = per_tok_logps  # (B*G, L-1)
                valid_mask_kl = valid_mask  # (B*G, L-1)
            # ---------------------------------------------------

            # Reference log-probs (frozen head)
            with torch.no_grad():
                # ensure device consistency
                self.ref_dense_head.to(device=pred_seqs.device, dtype=new_logps_kl.dtype)
                self.ref_dense_head.eval()
                ref_logps_kl, _, _ = self.ref_dense_head._get_token_logps_and_entropies(
                    batch_dict_exp, pred_seqs, self.tokenizer,
                    temperature=self.kl_temperature, compute_entropy=False
                )  # (B*G, L-1)


            # Sample-based KL estimate per token: E_{a~pi_new}[log pi_new(a) - log pi_ref(a)]
            kl_tokens = (new_logps_kl - ref_logps_kl)  # (B*G, L-1)
            # Per-sequence mean KL (mask invalid tokens)
            valid_counts = valid_mask_kl.sum(-1).clamp(min=1.0)         # (B*G,)
            kl_seq = (kl_tokens * valid_mask_kl).sum(-1) / valid_counts  # (B*G,)
            # Global token-mean KL
            kl_token_mean = (kl_tokens * valid_mask_kl).sum() / valid_mask_kl.sum().clamp(min=1.0)  # scalar


        # ---------------------------------------------------
        # compute loss
        # --------------------------------------------------
        log_ratio = per_tok_logps - old_per_tok_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio          # (B*G, L-1)
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * valid_mask).sum(-1) / valid_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)   # (B*G, 1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B*G, L-1); "sequence" level: (B*G, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)   # (B*G, L-1)

        # grpo loss
        loss = ((per_token_loss * valid_mask).sum(-1) / valid_mask.sum(-1).clamp(min=1.0)).mean()

        # keep a copy of PPO objective (pre-KL) for logging
        ppo_loss_for_log = loss.detach()

        # Add optional KL penalty (new || ref)
        kl_penalty = torch.tensor(0.0, device=loss.device)
        if (self.beta > 0) and (kl_seq is not None):
            if self.kl_penalty_level == 'token':
                kl_mean = kl_token_mean
            else:  # 'sequence'
                kl_mean = kl_seq.mean()
            kl_penalty = self.beta * kl_mean
            loss = loss + kl_penalty

        # Add optional entropy bonus
        entropy_bonus = torch.tensor(0.0, device=loss.device)
        if self.entropy_coeff > 0 and per_tok_entropies is not None:
            mean_entropy_for_bonus = (per_tok_entropies * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
            entropy_bonus = self.entropy_coeff * mean_entropy_for_bonus
            loss = loss - entropy_bonus  # maximize entropy

        # ---------------------------------------------------
        # logging
        # --------------------------------------------------
        valid_token_count = valid_mask.sum().clamp(min=1.0)
        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * valid_mask).sum() / valid_token_count

        mean_entropy = masked_batch_mean(per_tok_entropies)
        entropy_gather = concat_all_gather(mean_entropy[None]) if self.world_size > 1 else mean_entropy[None]

        # clipping ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped


        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        low_clip_gather = concat_all_gather(low_clip[None]) if self.world_size > 1 else low_clip[None]
        high_clip_gather = concat_all_gather(high_clip[None]) if self.world_size > 1 else high_clip[None]
        clip_ratio_gather = concat_all_gather(clip_ratio[None]) if self.world_size > 1 else clip_ratio[None]

        log_dict.update({
            'rl/entropy': torch.nanmean(entropy_gather).item(),
            'rl/clip_ratio/low_mean': torch.nanmean(low_clip_gather).item(),
            'rl/clip_ratio/low_min': nanmin(low_clip_gather).item(),
            'rl/clip_ratio/high_mean': torch.nanmean(high_clip_gather).item(),
            'rl/clip_ratio/high_max': nanmax(high_clip_gather).item(),
            'rl/clip_ratio/region_mean': torch.nanmean(clip_ratio_gather).item(),
        })

        # loss logging
        log_dict.update({
            'rl/agg_loss': all_reduce(loss.detach(), op='sum', average=True).item(),
            'rl/ppo_loss': all_reduce(ppo_loss_for_log, op='sum', average=True).item(),
        })

        # Gradient magnitude proxy: loss contributions split by advantage sign
        # Per-sample mean loss (proxy for gradient magnitude per sample)
        per_sample_loss = (per_token_loss * valid_mask).sum(-1) / valid_mask.sum(-1).clamp(min=1.0)  # (B*G,)
        pos_adv_mask = (advantages > 0)   # (B*G,)
        neg_adv_mask = (advantages < 0)   # (B*G,)
        if pos_adv_mask.any():
            pos_loss_mean = per_sample_loss[pos_adv_mask].mean()
            log_dict['rl/grad_proxy/pos_adv_loss'] = all_reduce(pos_loss_mean.detach(), op='sum', average=True).item()
            log_dict['rl/grad_proxy/pos_adv_count'] = all_reduce(pos_adv_mask.float().sum().detach(), op='sum', average=False).item()
        if neg_adv_mask.any():
            neg_loss_mean = per_sample_loss[neg_adv_mask].mean()
            log_dict['rl/grad_proxy/neg_adv_loss'] = all_reduce(neg_loss_mean.detach(), op='sum', average=True).item()
            log_dict['rl/grad_proxy/neg_adv_count'] = all_reduce(neg_adv_mask.float().sum().detach(), op='sum', average=False).item()
        # Fraction of samples with positive advantage
        log_dict['rl/grad_proxy/frac_pos_adv'] = all_reduce(pos_adv_mask.float().mean().detach(), op='sum', average=True).item()

        # KL logs (only when computed)
        if kl_seq is not None:
            # token-mean is scalar per GPU; average across GPUs for readability
            token_mean_g = all_reduce(kl_token_mean.detach(), op='sum', average=True)
            # sequence stats gathered across all ranks
            seq_g = concat_all_gather(kl_seq.detach()) if self.world_size > 1 else kl_seq.detach()
            log_dict.update({
                'rl/kl/token_mean': token_mean_g.item(),
                'rl/kl/seq_mean': torch.nanmean(seq_g).item(),
                'rl/kl/seq_min': nanmin(seq_g).item(),
                'rl/kl/seq_max': nanmax(seq_g).item(),
                'rl/kl_beta': float(self.beta),
                'rl/kl_penalty': kl_penalty.detach().item() if isinstance(kl_penalty, torch.Tensor) else float(kl_penalty),
            })

        if self.entropy_coeff > 0:
            log_dict['rl/entropy_bonus'] = entropy_bonus.detach().item()

        return loss, log_dict

    def simple_decode(self, batch_preds):
        """Simple decoding of predicted sequences without post-processing.

        Args:
            batch_preds: Tensor (B, L_pred) predicted token sequences including BOS
        Returns:
            pred_dicts: list of dicts, each containing:
                'pred_boxes': Tensor (N_boxes, 7) predicted boxes
                'pred_labels': Tensor (N_boxes,) predicted class labels
        """

        pred_dicts = []

        for index, pred_seq in enumerate(batch_preds):
            pred_cls, pred_boxes = self.tokenizer.decode(pred_seq)  # Nboxes, Nboxes x 7

            pred_cls[pred_cls < 0] = 0
            pred_cls[pred_cls > self.num_class] = 0

            #NOTE: No filtering of noise boxes here
            record_dict = {
                'pred_boxes': pred_boxes,
                'pred_labels': pred_cls,
            }
            pred_dicts.append(record_dict)

        return pred_dicts
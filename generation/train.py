####################################
#                                  #
#        import libraries          #
#                                  #
####################################


from src.utils import (
    create_logger,
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    load_config,
    base_config,
    build_dataloaders,
    validate_protein_dim_against_h5,
    a_useful_log,
    sample_t,
    build_xt_and_labels,
    build_random_negative_protein_batch,
    compute_diffusion_loss,
    compute_masked_token_ce,
    compute_pn_losses,
    save_model,
    resume_model,
)
import os
import math
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from src.model import EsmConfig, EsmForMaskedLM
from functools import partial
from transformers import AutoTokenizer
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from contextlib import nullcontext
from torch.nn.utils import clip_grad_norm_


####################################
#                                  #
#            Validate              #
#                                  #
####################################


@torch.no_grad()
def validate_one_epoch(
    model,
    val_loader,
    tokenizer,
    device,
    world_size: int,
    use_bf16: bool = True,
    t_eps: float = 1e-3,
) -> dict:
    model.eval()

    dataset = val_loader.dataset
    use_pn_objective = getattr(validate_one_epoch, "_use_pn_objective", False)
    pn_alpha = getattr(validate_one_epoch, "_pn_alpha", 0.2)
    pn_margin = getattr(validate_one_epoch, "_pn_margin", 0.2)
    pn_use_batch_negatives_first = getattr(
        validate_one_epoch,
        "_pn_use_batch_negatives_first",
        True,
    )
    val_rng = random.Random(0)

    if use_pn_objective:
        val_total_loss_sum = 0.0
        val_diff_loss_pos_sum = 0.0
        val_rank_loss_sum = 0.0
        val_sample_count = 0.0
        val_pos_masked_token_loss_sum = 0.0
        val_pos_masked_token_count = 0.0
        val_neg_masked_token_loss_sum = 0.0
        val_neg_masked_token_count = 0.0
        val_delta_ce_sum = 0.0
        val_margin_satisfied_sum = 0.0
    else:
        val_loss_sum = 0.0
        val_num_samples = 0
        val_masked_token_loss_sum = 0.0
        val_masked_token_count = 0.0

    for batch in val_loader:
        if use_pn_objective:
            neg_batch = build_random_negative_protein_batch(
                batch=batch,
                dataset=dataset,
                use_batch_negatives_first=pn_use_batch_negatives_first,
                rng=val_rng,
                deterministic=True,
            )

        x0 = batch["rna_input_ids_clean"].to(device, non_blocking=True)
        rna_attention_mask = batch["rna_attention_mask"].to(device, non_blocking=True)
        protein_cond = batch["protein_cond"].to(device, non_blocking=True)
        protein_attention_mask = batch["protein_attention_mask"].to(device, non_blocking=True)

        if use_pn_objective:
            neg_protein_cond = neg_batch["neg_protein_cond"].to(device, non_blocking=True)
            neg_protein_attention_mask = neg_batch["neg_protein_attention_mask"].to(
                device,
                non_blocking=True,
            )

        t = sample_t(
            batch_size=x0.size(0),
            device=device,
            eps=t_eps,
        )

        xt, labels, _ = build_xt_and_labels(
            x0=x0,
            rna_attention_mask=rna_attention_mask,
            mask_token_id=tokenizer.mask_token_id,
            t=t,
            special_token_ids=[
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
            ],
            force_at_least_one_mask=True,
        )

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else nullcontext()
        )

        with autocast_ctx:
            outputs_pos = model(
                protein_cond=protein_cond,
                protein_attention_mask=protein_attention_mask,
                input_ids=xt,
                attention_mask=rna_attention_mask,
            )
            logits_pos = outputs_pos.logits

            if use_pn_objective:
                outputs_neg = model(
                    protein_cond=neg_protein_cond,
                    protein_attention_mask=neg_protein_attention_mask,
                    input_ids=xt,
                    attention_mask=rna_attention_mask,
                )
                logits_neg = outputs_neg.logits
                pn_metrics = compute_pn_losses(
                    logits_pos=logits_pos,
                    logits_neg=logits_neg,
                    labels=labels,
                    t=t,
                    alpha=pn_alpha,
                    margin=pn_margin,
                    diffusion_eps=t_eps,
                )
            else:
                loss = compute_diffusion_loss(
                    logits=logits_pos,
                    labels=labels,
                    t=t,
                )

        if use_pn_objective:
            batch_size = float(x0.size(0))
            val_total_loss_sum += pn_metrics["total_loss"].detach().item() * batch_size
            val_diff_loss_pos_sum += pn_metrics["diff_loss_pos"].detach().item() * batch_size
            val_rank_loss_sum += pn_metrics["rank_loss_sum"].detach().item()
            val_sample_count += pn_metrics["sample_count"].detach().item()
            val_pos_masked_token_loss_sum += (
                pn_metrics["pos_masked_token_loss_sum"].detach().item()
            )
            val_pos_masked_token_count += (
                pn_metrics["pos_masked_token_count"].detach().item()
            )
            val_neg_masked_token_loss_sum += (
                pn_metrics["neg_masked_token_loss_sum"].detach().item()
            )
            val_neg_masked_token_count += (
                pn_metrics["neg_masked_token_count"].detach().item()
            )
            val_delta_ce_sum += pn_metrics["delta_ce_sum"].detach().item()
            val_margin_satisfied_sum += (
                pn_metrics["margin_satisfied_sum"].detach().item()
            )
        else:
            masked_token_ce, masked_token_count = compute_masked_token_ce(
                logits=logits_pos,
                labels=labels,
            )
            batch_size = x0.size(0)
            val_loss_sum += loss.detach().item() * batch_size
            val_num_samples += batch_size
            val_masked_token_loss_sum += (
                masked_token_ce.detach().item() * masked_token_count.detach().item()
            )
            val_masked_token_count += masked_token_count.detach().item()

    if use_pn_objective:
        val_stats = torch.tensor(
            [
                val_total_loss_sum,
                val_diff_loss_pos_sum,
                val_rank_loss_sum,
                val_sample_count,
                val_pos_masked_token_loss_sum,
                val_pos_masked_token_count,
                val_neg_masked_token_loss_sum,
                val_neg_masked_token_count,
                val_delta_ce_sum,
                val_margin_satisfied_sum,
            ],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1 and dist.is_initialized():
            dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
        return {
            "total_loss": (val_stats[0] / val_stats[3].clamp_min(1.0)).item(),
            "diff_loss_pos": (val_stats[1] / val_stats[3].clamp_min(1.0)).item(),
            "rank_loss": (val_stats[2] / val_stats[3].clamp_min(1.0)).item(),
            "masked_ce_pos": (val_stats[4] / val_stats[5].clamp_min(1.0)).item(),
            "masked_ce_neg": (val_stats[6] / val_stats[7].clamp_min(1.0)).item(),
            "delta_ce": (val_stats[8] / val_stats[3].clamp_min(1.0)).item(),
            "margin_satisfied_rate": (val_stats[9] / val_stats[3].clamp_min(1.0)).item(),
        }

    val_stats = torch.tensor(
        [
            val_loss_sum,
            val_num_samples,
            val_masked_token_loss_sum,
            val_masked_token_count,
        ],
        device=device,
        dtype=torch.float64,
    )
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(val_stats, op=dist.ReduceOp.SUM)
    return {
        "total_loss": (val_stats[0] / val_stats[1].clamp_min(1.0)).item(),
        "masked_ce_pos": (val_stats[2] / val_stats[3].clamp_min(1.0)).item(),
    }


####################################
#                                  #
#              train               #
#                                  #
####################################


def is_better_pn_checkpoint(current_metrics: dict, best_selection: dict | None) -> tuple[bool, dict]:
    current_selection = {
        "margin_satisfied_rate": float(current_metrics["margin_satisfied_rate"]),
        "delta_ce": float(current_metrics["delta_ce"]),
        "masked_ce_pos": float(current_metrics["masked_ce_pos"]),
    }

    if not best_selection:
        return True, current_selection

    current_tuple = (
        current_selection["margin_satisfied_rate"],
        current_selection["delta_ce"],
        -current_selection["masked_ce_pos"],
    )
    best_tuple = (
        float(best_selection.get("margin_satisfied_rate", float("-inf"))),
        float(best_selection.get("delta_ce", float("-inf"))),
        -float(best_selection.get("masked_ce_pos", float("inf"))),
    )
    return current_tuple > best_tuple, current_selection


def apply_optimizer_hparams(
    optimizer: torch.optim.Optimizer,
    *,
    lr: float,
    betas: tuple[float, float],
    weight_decay: float,
    eps: float,
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        param_group["betas"] = betas
        param_group["weight_decay"] = weight_decay
        param_group["eps"] = eps
        param_group.pop("initial_lr", None)


def train(rank: int, world_size: int, config: dict) -> None:

    # 1. setup multi GPU training
    port = config["train"]["port"]
    setup_ddp(rank, world_size, port)

    # 2. try...except...finally...
    try:

        # 3. logger
        log_file = config["train"]["log_file"]
        logger = create_logger(log_file, rank)
        if is_main_process(rank):
            logger.info(f"Start Training! world_size={world_size}")
        log = partial(a_useful_log, logger, rank)

        project_root = os.path.dirname(os.path.abspath(__file__))
        protein_dim = validate_protein_dim_against_h5(config, base_dir=project_root)
        log(f"protein_dim={protein_dim} (verified against H5)")

        # 4. dataset
        train_loader, test_loader, train_sampler, test_sampler, tokenizer =build_dataloaders(
            config=config,
            world_size=world_size,
            rank=rank,
        )
        train_dataset = train_loader.dataset
        val_dataset = test_loader.dataset
        log(f"train samples={len(train_dataset)}, val samples={len(val_dataset)}")
        batch_size_per_gpu = config["train"]["batch_size_per_gpu"]
        grad_accum_steps = config["train"].get("grad_accum_steps", 1)
        effective_batch_size = batch_size_per_gpu * world_size * grad_accum_steps
        log("DataLoaders created.")
        log(f"batch_size_per_gpu={batch_size_per_gpu}")
        log(f"grad_accum_steps={grad_accum_steps}")
        log(f"effective_batch_size={effective_batch_size}")

        if effective_batch_size != 32:
            log(
                f"Warning: effective_batch_size={effective_batch_size} "
                "(batch_size_per_gpu × GPUs × grad_accum_steps)."
            )

        # 5. model
        model_config = EsmConfig(**base_config)
        model_config.vocab_size = len(tokenizer)
        if tokenizer.mask_token_id is not None:
            model_config.mask_token_id = tokenizer.mask_token_id
        if tokenizer.pad_token_id is not None:
            model_config.pad_token_id = tokenizer.pad_token_id
        model_config.use_FiLM = config["model"]["use_FiLM"]
        model_config.use_AdaLN = config["model"]["use_AdaLN"]
        model_config.use_gated_bias = config["model"]["use_gated_bias"]
        model_config.use_protein_conditioning_attention = config["model"]["use_protein_conditioning_attention"]
        model_config.protein_dim = config["model"]["protein_dim"]
        if config["model"].get("num_hidden_layers") is not None:
            model_config.num_hidden_layers = int(config["model"]["num_hidden_layers"])

        model = EsmForMaskedLM(model_config)
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )
        log(
            f"Model initialized. vocab_size={model_config.vocab_size}, "
            f"mask_token_id={model_config.mask_token_id}, pad_token_id={model_config.pad_token_id}"
        )

        use_pn_objective = config["train"].get("use_pn_objective", True)
        pn_alpha = float(config["train"].get("pn_alpha", 0.2))
        pn_margin = float(config["train"].get("pn_margin", 0.2))
        pn_num_negatives = int(config["train"].get("pn_num_negatives", 1))
        pn_sampling_strategy = config["train"].get("pn_sampling_strategy", "random")
        lr_schedule = config["train"].get("lr_schedule", "constant")
        pn_use_batch_negatives_first = config["train"].get(
            "pn_use_batch_negatives_first",
            True,
        )

        if use_pn_objective and pn_num_negatives != 1:
            raise NotImplementedError(
                f"Only pn_num_negatives=1 is implemented for now, got {pn_num_negatives}."
            )
        if use_pn_objective and pn_sampling_strategy != "random":
            raise NotImplementedError(
                f"Only pn_sampling_strategy='random' is implemented for now, got {pn_sampling_strategy}."
            )
        if lr_schedule != "constant":
            raise NotImplementedError(
                f"Only lr_schedule='constant' is supported, got {lr_schedule}."
            )

        # 6. optimizer
        optimizer_betas = tuple(config["train"].get("betas", [0.9, 0.95]))
        if len(optimizer_betas) != 2:
            raise ValueError(
                f"`train.betas` must contain exactly two values, got {optimizer_betas}."
            )
        base_lr = float(config["train"].get("lr", 8e-5))
        weight_decay = float(config["train"].get("weight_decay", 0.01))
        adam_eps = float(config["train"].get("adam_eps", 1e-8))

        optimizer = AdamW(
            model.parameters(),
            lr=base_lr,
            betas=optimizer_betas,
            weight_decay=weight_decay,
            eps=adam_eps,
        )
        num_epochs = config["train"]["num_epochs"]
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        total_update_steps = steps_per_epoch * num_epochs
        scheduler = None

        log(f"steps_per_epoch={steps_per_epoch}")
        log(f"total_update_steps={total_update_steps}")
        log(f"lr={base_lr}")
        log(f"betas={optimizer_betas}")
        log(f"weight_decay={weight_decay}")
        log(f"adam_eps={adam_eps}")
        log(f"lr_schedule={lr_schedule}")
        log(f"use_pn_objective={use_pn_objective}")
        if use_pn_objective:
            log(f"pn_alpha={pn_alpha}")
            log(f"pn_margin={pn_margin}")
            log(f"pn_num_negatives={pn_num_negatives}")
            log(f"pn_sampling_strategy={pn_sampling_strategy}")
            log(f"pn_use_batch_negatives_first={pn_use_batch_negatives_first}")


        # 6.5 resume
        best_val_loss = float("inf")
        best_val_selection = None
        start_epoch = 0
        global_step = 0

        resume_path = config["train"].get("resume_path", None)
        if resume_path:
            resume_state = resume_model(
                load_path=resume_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                strict=True,
            )
            start_epoch = resume_state["epoch"] + 1
            global_step = resume_state["global_step"]
            best_val_loss = resume_state["extra_state"].get("best_val_loss",float("inf"))
            best_val_selection = resume_state["extra_state"].get("best_val_selection", None)
            apply_optimizer_hparams(
                optimizer,
                lr=base_lr,
                betas=optimizer_betas,
                weight_decay=weight_decay,
                eps=adam_eps,
            )
            log(
                f"Resumed from {resume_path} | "
                f"start_epoch={start_epoch}, global_step={global_step}, lr={base_lr:.6e}"
            )


        # 7. training loop
        use_bf16 = config["train"].get("use_bf16", True)
        log_every = config["train"].get("log_every", 50)
        clip_max_norm = config["train"].get("grad_clip", 1.0)
        t_eps = config["train"].get("t_eps", 1e-3)

        optimizer.zero_grad(set_to_none=True)
        ema_log_total_loss = None
        ema_log_masked_token_ce = None
        ema_log_rank_loss = None

        for epoch in range(start_epoch, num_epochs):

            # 7.1 set epoch for distributed sampler
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # 7.2 train one epoch
            model.train()
            num_train_batches = len(train_loader)
            last_accum_steps = num_train_batches % grad_accum_steps
            if last_accum_steps == 0:
                last_accum_steps = grad_accum_steps

            if use_pn_objective:
                epoch_total_loss_sum = 0.0
                epoch_diff_loss_pos_sum = 0.0
                epoch_rank_loss_sum = 0.0
                epoch_sample_count = 0.0
                epoch_pos_masked_token_loss_sum = 0.0
                epoch_pos_masked_token_count = 0.0
                epoch_neg_masked_token_loss_sum = 0.0
                epoch_neg_masked_token_count = 0.0
                epoch_delta_ce_sum = 0.0
                epoch_margin_satisfied_sum = 0.0

                step_total_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
                step_diff_loss_pos_sum = torch.zeros((), device=device, dtype=torch.float64)
                step_rank_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
                step_sample_count = torch.zeros((), device=device, dtype=torch.float64)
                step_pos_masked_token_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
                step_pos_masked_token_count = torch.zeros((), device=device, dtype=torch.float64)
                step_neg_masked_token_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
                step_neg_masked_token_count = torch.zeros((), device=device, dtype=torch.float64)
                step_delta_ce_sum = torch.zeros((), device=device, dtype=torch.float64)
                step_margin_satisfied_sum = torch.zeros((), device=device, dtype=torch.float64)
            else:
                epoch_loss_sum = 0.0
                epoch_num_batches = 0
                epoch_masked_token_loss_sum = 0.0
                epoch_masked_token_count = 0.0

                step_masked_token_loss_sum = torch.zeros(
                    (),
                    device=device,
                    dtype=torch.float64,
                )
                step_masked_token_count = torch.zeros(
                    (),
                    device=device,
                    dtype=torch.float64,
                )

            for step, batch in enumerate(train_loader):
                if use_pn_objective:
                    neg_batch = build_random_negative_protein_batch(
                        batch=batch,
                        dataset=train_dataset,
                        use_batch_negatives_first=pn_use_batch_negatives_first,
                    )

                # 7.2.1 move batch to device
                x0 = batch["rna_input_ids_clean"].to(device, non_blocking=True)
                rna_attention_mask = batch["rna_attention_mask"].to(device, non_blocking=True)
                protein_cond = batch["protein_cond"].to(device, non_blocking=True)
                protein_attention_mask = batch["protein_attention_mask"].to(device, non_blocking=True)
                if use_pn_objective:
                    neg_protein_cond = neg_batch["neg_protein_cond"].to(device, non_blocking=True)
                    neg_protein_attention_mask = neg_batch["neg_protein_attention_mask"].to(
                        device,
                        non_blocking=True,
                    )

                # 7.2.2 sample t
                t = sample_t(
                    batch_size=x0.size(0),
                    device=device,
                    eps=t_eps,
                )

                # 7.2.3 build x_t and labels
                xt, labels, masked_positions = build_xt_and_labels(
                    x0=x0,
                    rna_attention_mask=rna_attention_mask,
                    mask_token_id=tokenizer.mask_token_id,
                    t=t,
                    special_token_ids=[
                tokenizer.pad_token_id,
                tokenizer.cls_token_id,
            ],
                    force_at_least_one_mask=True,
                )

                # 7.2.4 forward + diffusion loss
                autocast_ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16
                    else nullcontext()
                )

                with autocast_ctx:
                    outputs_pos = model(
                        protein_cond=protein_cond,
                        protein_attention_mask=protein_attention_mask,
                        input_ids=xt,
                        attention_mask=rna_attention_mask,
                    )
                    logits_pos = outputs_pos.logits
                    if use_pn_objective:
                        outputs_neg = model(
                            protein_cond=neg_protein_cond,
                            protein_attention_mask=neg_protein_attention_mask,
                            input_ids=xt,
                            attention_mask=rna_attention_mask,
                        )
                        logits_neg = outputs_neg.logits
                        pn_metrics = compute_pn_losses(
                            logits_pos=logits_pos,
                            logits_neg=logits_neg,
                            labels=labels,
                            t=t,
                            alpha=pn_alpha,
                            margin=pn_margin,
                            diffusion_eps=t_eps,
                        )
                        loss = pn_metrics["total_loss"]
                    else:
                        loss = compute_diffusion_loss(
                            logits=logits_pos,
                            labels=labels,
                            t=t,
                        )

                if use_pn_objective:
                    batch_size = float(x0.size(0))
                    loss_for_log = loss.detach()

                    epoch_total_loss_sum += loss_for_log.item() * batch_size
                    epoch_diff_loss_pos_sum += (
                        pn_metrics["diff_loss_pos"].detach().item() * batch_size
                    )
                    epoch_rank_loss_sum += pn_metrics["rank_loss_sum"].detach().item()
                    epoch_sample_count += pn_metrics["sample_count"].detach().item()
                    epoch_pos_masked_token_loss_sum += (
                        pn_metrics["pos_masked_token_loss_sum"].detach().item()
                    )
                    epoch_pos_masked_token_count += (
                        pn_metrics["pos_masked_token_count"].detach().item()
                    )
                    epoch_neg_masked_token_loss_sum += (
                        pn_metrics["neg_masked_token_loss_sum"].detach().item()
                    )
                    epoch_neg_masked_token_count += (
                        pn_metrics["neg_masked_token_count"].detach().item()
                    )
                    epoch_delta_ce_sum += pn_metrics["delta_ce_sum"].detach().item()
                    epoch_margin_satisfied_sum += (
                        pn_metrics["margin_satisfied_sum"].detach().item()
                    )

                    step_total_loss_sum += loss_for_log.to(torch.float64) * batch_size
                    step_diff_loss_pos_sum += (
                        pn_metrics["diff_loss_pos"].detach().to(torch.float64) * batch_size
                    )
                    step_rank_loss_sum += pn_metrics["rank_loss_sum"].detach()
                    step_sample_count += pn_metrics["sample_count"].detach()
                    step_pos_masked_token_loss_sum += (
                        pn_metrics["pos_masked_token_loss_sum"].detach()
                    )
                    step_pos_masked_token_count += (
                        pn_metrics["pos_masked_token_count"].detach()
                    )
                    step_neg_masked_token_loss_sum += (
                        pn_metrics["neg_masked_token_loss_sum"].detach()
                    )
                    step_neg_masked_token_count += (
                        pn_metrics["neg_masked_token_count"].detach()
                    )
                    step_delta_ce_sum += pn_metrics["delta_ce_sum"].detach()
                    step_margin_satisfied_sum += (
                        pn_metrics["margin_satisfied_sum"].detach()
                    )
                else:
                    masked_token_ce, masked_token_count = compute_masked_token_ce(
                        logits=logits_pos,
                        labels=labels,
                    )
                    masked_token_count = masked_token_count.detach().to(torch.float64)
                    masked_token_loss_sum = (
                        masked_token_ce.detach().to(torch.float64) * masked_token_count
                    )

                    loss_for_log = loss.detach()
                    epoch_loss_sum += loss_for_log.item()
                    epoch_num_batches += 1
                    epoch_masked_token_loss_sum += masked_token_loss_sum.item()
                    epoch_masked_token_count += masked_token_count.item()
                    step_masked_token_loss_sum += masked_token_loss_sum
                    step_masked_token_count += masked_token_count

                # 7.2.5 backward
                in_last_window = step >= num_train_batches - last_accum_steps
                current_accum_steps = last_accum_steps if in_last_window else grad_accum_steps

                loss = loss / current_accum_steps
                loss.backward()

                # 7.2.6 optimizer step
                should_step = (
                    ((step + 1) % grad_accum_steps == 0)
                    or ((step + 1) == len(train_loader))
                )

                if should_step:
                    grad_norm = clip_grad_norm_(model.parameters(),max_norm=clip_max_norm)

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1

                    if use_pn_objective:
                        step_stats = torch.stack(
                            [
                                step_total_loss_sum,
                                step_diff_loss_pos_sum,
                                step_rank_loss_sum,
                                step_sample_count,
                                step_pos_masked_token_loss_sum,
                                step_pos_masked_token_count,
                                step_neg_masked_token_loss_sum,
                                step_neg_masked_token_count,
                                step_delta_ce_sum,
                                step_margin_satisfied_sum,
                            ]
                        )
                        if world_size > 1 and dist.is_initialized():
                            dist.all_reduce(step_stats, op=dist.ReduceOp.SUM)

                        log_total_loss = (
                            step_stats[0] / step_stats[3].clamp_min(1.0)
                        ).item()
                        log_diff_loss_pos = (
                            step_stats[1] / step_stats[3].clamp_min(1.0)
                        ).item()
                        log_rank_loss = (
                            step_stats[2] / step_stats[3].clamp_min(1.0)
                        ).item()
                        log_masked_ce_pos = (
                            step_stats[4] / step_stats[5].clamp_min(1.0)
                        ).item()
                        log_masked_ce_neg = (
                            step_stats[6] / step_stats[7].clamp_min(1.0)
                        ).item()
                        log_delta_ce = (
                            step_stats[8] / step_stats[3].clamp_min(1.0)
                        ).item()
                        log_margin_rate = (
                            step_stats[9] / step_stats[3].clamp_min(1.0)
                        ).item()

                        step_total_loss_sum.zero_()
                        step_diff_loss_pos_sum.zero_()
                        step_rank_loss_sum.zero_()
                        step_sample_count.zero_()
                        step_pos_masked_token_loss_sum.zero_()
                        step_pos_masked_token_count.zero_()
                        step_neg_masked_token_loss_sum.zero_()
                        step_neg_masked_token_count.zero_()
                        step_delta_ce_sum.zero_()
                        step_margin_satisfied_sum.zero_()

                        if is_main_process(rank) and (
                            global_step == 1 or global_step % log_every == 0
                        ):
                            if ema_log_total_loss is None:
                                ema_log_total_loss = log_total_loss
                            else:
                                ema_log_total_loss = (
                                    0.98 * ema_log_total_loss + 0.02 * log_total_loss
                                )
                            if ema_log_masked_token_ce is None:
                                ema_log_masked_token_ce = log_masked_ce_pos
                            else:
                                ema_log_masked_token_ce = (
                                    0.98 * ema_log_masked_token_ce + 0.02 * log_masked_ce_pos
                                )
                            if ema_log_rank_loss is None:
                                ema_log_rank_loss = log_rank_loss
                            else:
                                ema_log_rank_loss = (
                                    0.98 * ema_log_rank_loss + 0.02 * log_rank_loss
                                )
                            log(
                                f"Epoch [{epoch + 1}/{num_epochs}] "
                                f"Step [{global_step}/{total_update_steps}] "
                                f"LossEMA: {ema_log_total_loss:.4f} "
                                f"MaskedTokenCEEMA: {ema_log_masked_token_ce:.4f} "
                                f"RankLossEMA: {ema_log_rank_loss:.4f}"
                            )
                    else:
                        log_loss = loss_for_log.clone()
                        if world_size > 1 and dist.is_initialized():
                            dist.all_reduce(log_loss, op=dist.ReduceOp.SUM)
                            log_loss = log_loss / world_size

                        step_masked_token_stats = torch.stack(
                            [step_masked_token_loss_sum, step_masked_token_count]
                        )
                        if world_size > 1 and dist.is_initialized():
                            dist.all_reduce(step_masked_token_stats, op=dist.ReduceOp.SUM)
                        log_masked_token_ce = (
                            step_masked_token_stats[0]
                            / step_masked_token_stats[1].clamp_min(1.0)
                        ).item()
                        step_masked_token_loss_sum.zero_()
                        step_masked_token_count.zero_()

                        if is_main_process(rank) and (
                            global_step == 1 or global_step % log_every == 0
                        ):
                            loss_scalar = log_loss.item()
                            if ema_log_total_loss is None:
                                ema_log_total_loss = loss_scalar
                            else:
                                ema_log_total_loss = (
                                    0.98 * ema_log_total_loss + 0.02 * loss_scalar
                                )
                            if ema_log_masked_token_ce is None:
                                ema_log_masked_token_ce = log_masked_token_ce
                            else:
                                ema_log_masked_token_ce = (
                                    0.98 * ema_log_masked_token_ce + 0.02 * log_masked_token_ce
                                )
                            log(
                                f"Epoch [{epoch + 1}/{num_epochs}] "
                                f"Step [{global_step}/{total_update_steps}] "
                                f"LossEMA: {ema_log_total_loss:.4f} "
                                f"MaskedTokenCEEMA: {ema_log_masked_token_ce:.4f}"
                            )

            # 7.3 epoch summary
            if use_pn_objective:
                epoch_stats = torch.tensor(
                    [
                        epoch_total_loss_sum,
                        epoch_diff_loss_pos_sum,
                        epoch_rank_loss_sum,
                        epoch_sample_count,
                        epoch_pos_masked_token_loss_sum,
                        epoch_pos_masked_token_count,
                        epoch_neg_masked_token_loss_sum,
                        epoch_neg_masked_token_count,
                        epoch_delta_ce_sum,
                        epoch_margin_satisfied_sum,
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                if world_size > 1 and dist.is_initialized():
                    dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)

                mean_epoch_total_loss = (epoch_stats[0] / epoch_stats[3].clamp_min(1.0)).item()
                mean_epoch_diff_loss_pos = (epoch_stats[1] / epoch_stats[3].clamp_min(1.0)).item()
                mean_epoch_rank_loss = (epoch_stats[2] / epoch_stats[3].clamp_min(1.0)).item()
                mean_epoch_masked_ce_pos = (epoch_stats[4] / epoch_stats[5].clamp_min(1.0)).item()
                mean_epoch_masked_ce_neg = (epoch_stats[6] / epoch_stats[7].clamp_min(1.0)).item()
                mean_epoch_delta_ce = (epoch_stats[8] / epoch_stats[3].clamp_min(1.0)).item()
                mean_epoch_margin_rate = (epoch_stats[9] / epoch_stats[3].clamp_min(1.0)).item()

                if is_main_process(rank):
                    log(
                        f"Epoch [{epoch + 1}/{num_epochs}] finished. "
                        f"MeanLoss: {mean_epoch_total_loss:.4f} "
                        f"MeanMaskedTokenCE: {mean_epoch_masked_ce_pos:.4f} "
                        f"MeanRankLoss: {mean_epoch_rank_loss:.4f}"
                    )
            else:
                epoch_stats = torch.tensor(
                    [
                        epoch_loss_sum,
                        epoch_num_batches,
                        epoch_masked_token_loss_sum,
                        epoch_masked_token_count,
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                if world_size > 1 and dist.is_initialized():
                    dist.all_reduce(epoch_stats, op=dist.ReduceOp.SUM)

                mean_epoch_loss = (epoch_stats[0] / epoch_stats[1]).item()
                mean_epoch_masked_token_ce = (
                    epoch_stats[2] / epoch_stats[3].clamp_min(1.0)
                ).item()

                if is_main_process(rank):
                    log(
                        f"Epoch [{epoch + 1}/{num_epochs}] finished. "
                        f"MeanLoss: {mean_epoch_loss:.4f} "
                        f"MeanMaskedTokenCE: {mean_epoch_masked_token_ce:.4f}"
                    )

            # 7.4 validation
            validate_one_epoch._use_pn_objective = use_pn_objective
            validate_one_epoch._pn_alpha = pn_alpha
            validate_one_epoch._pn_margin = pn_margin
            validate_one_epoch._pn_use_batch_negatives_first = (
                pn_use_batch_negatives_first
            )
            val_metrics = validate_one_epoch(
                model=model,
                val_loader=test_loader,
                tokenizer=tokenizer,
                device=device,
                world_size=world_size,
                use_bf16=use_bf16,
                t_eps=t_eps,
            )

            if use_pn_objective:
                is_best, updated_selection = is_better_pn_checkpoint(
                    current_metrics=val_metrics,
                    best_selection=best_val_selection,
                )
                best_val_loss = min(best_val_loss, val_metrics["total_loss"])
                if is_best:
                    best_val_selection = updated_selection

                if is_main_process(rank):
                    log(
                        f"Epoch [{epoch + 1}/{num_epochs}] "
                        f"Validation Loss: {val_metrics['total_loss']:.4f} "
                        f"Validation MaskedTokenCE: {val_metrics['masked_ce_pos']:.4f} "
                        f"Validation RankLoss: {val_metrics['rank_loss']:.4f}"
                    )
            else:
                is_best = val_metrics["total_loss"] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics["total_loss"]

                if is_main_process(rank):
                    log(
                        f"Epoch [{epoch + 1}/{num_epochs}] "
                        f"Validation Loss: {val_metrics['total_loss']:.4f} "
                        f"Validation MaskedTokenCE: {val_metrics['masked_ce_pos']:.4f}"
                    )

            if is_main_process(rank):
                ckpt_dir = config["train"]["ckpt_dir"]
                extra_state = {"best_val_loss": best_val_loss}
                if use_pn_objective:
                    extra_state["best_val_selection"] = best_val_selection

                save_model(
                    save_path=os.path.join(ckpt_dir, "latest.pt"),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                    extra_state=extra_state,
                )
                if is_best:
                    save_model(
                        save_path=os.path.join(ckpt_dir, "best.pt"),
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        config=config,
                        extra_state=extra_state,
                    )
                save_model(
                    save_path=os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt"),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                    extra_state=extra_state,
                )
                if is_best:
                    log(f"Checkpoint saved for epoch {epoch + 1} | latest + best")
                else:
                    log(f"Checkpoint saved for epoch {epoch + 1} | latest only")
            if world_size > 1 and dist.is_initialized():
                dist.barrier()

        if world_size > 1 and dist.is_initialized():
            dist.barrier()
        log("Training completed.")
    except Exception:
        if "logger" in locals():
            logger.exception("Training crashed with an exception.")
        raise

    finally:
        cleanup_ddp()


####################################
#                                  #
#              main()              #
#                                  #
####################################


def main(config):

    config = load_config(config)

    # GPU env
    gpu_ids = config["train"]["gpu_ids"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # num GPUs
    world_size = len(config["train"]["gpu_ids"])
    if world_size >= 2:
        mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)
    elif world_size == 1:
        train(rank = 0, world_size = 1, config = config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ProRiboGen training")
    parser.add_argument(
        "--config",
        default="config/train.json",
        help="Path to training config JSON",
    )
    args = parser.parse_args()
    main(args.config)

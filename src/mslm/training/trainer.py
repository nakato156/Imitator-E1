from tqdm import tqdm
import os
import typing as t

from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

torch.manual_seed(23)
random.seed(23)

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.checkpoint.manager import CheckpointManager
# from src.mslm.training import imitator_loss
from src.mslm.training.loss_msepcossim import imitator_loss
from src.mslm.training.loss_ce_vocab import imitator_ce_loss, ce_floor
from src.mslm.training.loss_ce_ar import imitator_ar_loss, build_ar_labels as _build_ar_labels
from src.mslm.training.collapse_metrics import compute_collapse_metrics
from src.mslm.training.sigreg import sigreg_loss
from src.mslm.training.loss_infonce import prefix_text_infonce
import nvtx
from datetime import datetime

def _char_ngrams(text: str, n: int) -> "dict":
    from collections import Counter
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _compute_chrf(hypotheses: "list[str]", references: "list[str]") -> float:
    """Compute corpus-level chrF score (0–100). Uses sacrebleu if available."""
    try:
        from sacrebleu.metrics import CHRF
        return CHRF().corpus_score(hypotheses, [references]).score
    except ImportError:
        pass
    # Fallback: character bigram F-score
    total_p, total_r, n = 0.0, 0.0, 0
    for hyp, ref in zip(hypotheses, references):
        hyp_bg = _char_ngrams(hyp, 2)
        ref_bg = _char_ngrams(ref, 2)
        if not ref_bg:
            continue
        matches = sum((hyp_bg & ref_bg).values())
        p = matches / max(sum(hyp_bg.values()), 1)
        r = matches / max(sum(ref_bg.values()), 1)
        total_p += p
        total_r += r
        n += 1
    if n == 0:
        return 0.0
    p, r = total_p / n, total_r / n
    return 100.0 * (2 * p * r) / (p + r) if (p + r) > 0 else 0.0


class Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate, save_tb_model=True, **kwargs):
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
            dynamic=True
        )

        #Accelerator module
        self.accelerator = Accelerator(mixed_precision="bf16", dynamo_plugin=dynamo_plugin)
        self.device = self.accelerator.device

        #Hyperparameters
        self.epochs = kwargs.get("epochs", 100)
        self.learning_rate = learning_rate

        #Loggers
        self.log_interval = kwargs.get("log_interval", 5)
        self.save_tb_model = save_tb_model

        version = kwargs.get("model_version", 1)
        checkpoint = kwargs.get("checkpoint", 1)

        self.writer = SummaryWriter(f"../outputs/reports/{version}/{checkpoint}/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
        self.graph_added = False
        
        #Save and checkpoint
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 5)
        self.ckpt_mgr = CheckpointManager(
            kwargs.get("model_dir", "../outputs/checkpoints"),
            version,
            checkpoint,
        )

        #Loss Function
        # Experimento CE-vocab (v115): [loss] type = "ce_vocab" clasifica cada posición
        # contra el vocabulario del LLM (cabeza atada a la tabla de embeddings) en vez de
        # regresar el embedding con MSE+coseno (que colapsa a la media condicional).
        loss_cfg = kwargs.get("loss")
        if loss_cfg is None:
            try:
                from src.mslm.utils.config_loader import cfg
                loss_cfg = dict(getattr(cfg, "loss", {}))
            except Exception:
                loss_cfg = {}
        self.loss_type = loss_cfg.get("type", "mse_cossim")

        self.embed_table = None
        self.embed_table_norm = None
        self.log_temp = None
        self.bridge = None
        if self.loss_type == "ce_vocab":
            table_path = loss_cfg["embed_table_path"]
            self.embed_table = torch.load(table_path, map_location="cpu").float().to(self.device)
            self.embed_table.requires_grad_(False)
            self.embed_table_norm = F.normalize(self.embed_table, dim=-1)
            self.embed_table_norm.requires_grad_(False)

            self.label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

            use_learnable_temp = bool(loss_cfg.get("learnable_temp", False))
            if use_learnable_temp:
                init_log_temp = float(loss_cfg.get("init_log_temp", 2.659))
                self.log_temp = nn.Parameter(torch.tensor(init_log_temp, device=self.device))
                logit_temp_arg = self.log_temp
            else:
                logit_temp_arg = float(loss_cfg.get("logit_temp", 1.0))

            print(f"CE-vocab: tabla {tuple(self.embed_table.shape)} desde {table_path} | learnable_temp={use_learnable_temp} label_smoothing={self.label_smoothing}")
            base_criterion = lambda output, embedding, mask_embedding, token_ids: imitator_ce_loss(
                output, token_ids, self.embed_table_norm, logit_temp_arg, self.label_smoothing
            )
        elif self.loss_type == "ce_ar":
            llm_model = loss_cfg["llm_model"]
            self.k_prefix = int(loss_cfg.get("k_prefix", 20))
            # lazy import avoids circular import at module load time
            from src.mslm.models.gemma_bridge import GemmaBridge
            self.bridge = GemmaBridge(llm_model)
            print(f"CE-AR v116: GemmaBridge {llm_model} | k_prefix={self.k_prefix} | hidden={self.bridge.hidden_size} vocab={self.bridge.vocab_size}")

            _k = self.k_prefix
            _bridge = self.bridge
            _self = self

            def _ce_ar_criterion(output, embedding, mask_embedding, token_ids):
                B, L = token_ids.shape
                # embed target tokens for teacher-forcing input (tokens 0..L-2)
                text_ids_orig = token_ids[:, :-1]                    # [B, L-1] — keep -100
                text_ids = text_ids_orig.clone()
                text_ids[text_ids == -100] = 0  # safe embedding lookup for padding positions
                text_embeds = _bridge.embed_tokens(text_ids)        # [B, L-1, H]
                inputs_embeds = torch.cat([output, text_embeds], dim=1)  # [B, K+L-1, H]
                # Bug-3 fix: mask padding tokens so they don't attend (causal, minimal impact
                # but cleaner). Prefix positions always attend (they are not padding).
                text_valid = (token_ids[:, :-1] != -100).long()         # [B, L-1]
                attn_mask = torch.cat([
                    torch.ones(B, _k, dtype=torch.long, device=output.device),
                    text_valid,
                ], dim=1)                                               # [B, K+L-1]
                logits = _bridge.forward(inputs_embeds, attn_mask)      # [B, K+L-1, V]
                # Bug-1 fix: mask only K-1 prefix positions so that position K-1
                # (last prefix slot) supervises token_ids[:, 0] (first text token).
                labels = _build_ar_labels(token_ids, _k)               # [B, K+L-1]
                ce, _, top1, top5 = imitator_ar_loss(logits, labels, _k)
                if _self.infonce_enabled:
                    nce = prefix_text_infonce(output, text_embeds, text_ids_orig, _self.infonce_temperature)
                    ce_total = ce + _self.infonce_lambda * nce
                    _self._infonce_sum += nce.detach().item()
                    _self._infonce_n += 1
                else:
                    ce_total = ce
                return ce_total, ce.detach(), top1, top5

            base_criterion = _ce_ar_criterion
        else:
            base_criterion = lambda output, embedding, mask_embedding, token_ids: imitator_loss(
                output, embedding, mask_embedding
            )

        if kwargs.get("compile", True):
            self.criterion = torch.compile(
                base_criterion,
                backend="inductor",
                mode="default",
                dynamic=True
            )
        else:
            self.criterion = base_criterion

        #Model
        self.model = model
        self.load_previous_model = kwargs.get("load_previous_model", False)
                
        #Dataloaders
        self.train_loader = self.accelerator.prepare_data_loader(train_loader)
        self.val_loader = self.accelerator.prepare_data_loader(val_loader)

        #Stopper
        es_patience = int(kwargs.get("early_stopping_patience", 100))
        self.early_stopping = EarlyStopping(patience=es_patience)

        #Optimizer
        self.optimizer = None
        self.scheduler = None

        #Batch Sampling
        self.batch_size = kwargs.get("batch_size", 5)
        self.batch_sampling = kwargs.get("batch_sampling", True)
        if self.batch_sampling:
            self.sub_batch = kwargs.get("batch_sample", 4)

        #Options 
        self.prof = False
        self.distributed = None
        
        self.grad_clip = kwargs.get("grad_clip", 0.1)
        self.weight_decay = kwargs.get("weight_decay", 0.05)

        #Diagnóstico de colapso de embeddings (Fase 1 del experimento SIGReg).
        diag = kwargs.get("diagnostics")
        if diag is None:
            try:
                from src.mslm.utils.config_loader import cfg
                diag = getattr(cfg, "diagnostics", {})
            except Exception:
                diag = {}
        self.diagnostics_enabled = bool(diag.get("enabled", False))
        self.diag_interval = int(diag.get("eval_interval", 1))
        self.diag_n_batches = int(diag.get("n_eval_batches", 16))
        self.diag_compare_target = bool(diag.get("compare_to_target", True))
        self.diag_eps = float(diag.get("per_dim_std_eps", 0.01))
        self.diag_alert_ratio = float(diag.get("collapse_alert_effrank_ratio", 0.0))

        # SIGReg (Fase 2): regularizador anti-colapso hacia gaussiana isotrópica (LeJEPA).
        sg = kwargs.get("sigreg")
        if sg is None:
            try:
                from src.mslm.utils.config_loader import cfg
                sg = getattr(cfg, "sigreg", {})
            except Exception:
                sg = {}
        self.sigreg_enabled = bool(sg.get("enabled", False))
        self.sigreg_lambda = float(sg.get("lambda", 1.0))
        self.sigreg_kwargs = dict(
            n_slices=int(sg.get("n_slices", 1024)),
            n_freqs=int(sg.get("n_freqs", 17)),
            standardize=bool(sg.get("standardize", True)),
            resample_slices=bool(sg.get("resample_slices", True)),
        )
        self._sigreg_sum = 0.0
        self._sigreg_n = 0

        # InfoNCE contrastive loss (v117): prevents prefix collapse.
        infonce_cfg = kwargs.get("infonce")
        if infonce_cfg is None:
            try:
                from src.mslm.utils.config_loader import cfg
                infonce_cfg = dict(getattr(cfg, "infonce", {}))
            except Exception:
                infonce_cfg = {}
        self.infonce_enabled = bool(infonce_cfg.get("enabled", False))
        self.infonce_lambda = float(infonce_cfg.get("lambda", 0.1))
        self.infonce_temperature = float(infonce_cfg.get("temperature", 0.07))
        self._infonce_sum = 0.0
        self._infonce_n = 0

        # Gradient norm logging (v117).
        self._grad_norm_sum = 0.0
        self._grad_norm_n = 0

        # Best checkpoint by chrF (v117) — selected in _run_generation_eval.
        self.best_chrf = -float("inf")

        # Generative evaluation (v116.0): greedy decode from soft prefix + chrF.
        gen_cfg = kwargs.get("eval_gen")
        if gen_cfg is None:
            try:
                from src.mslm.utils.config_loader import cfg
                gen_cfg = dict(getattr(cfg, "eval", {}))
            except Exception:
                gen_cfg = {}
        self.gen_interval = int(gen_cfg.get("gen_interval", 5))
        self.gen_max_tokens = int(gen_cfg.get("max_new_tokens", 50))
        self.gen_val_batches = int(gen_cfg.get("gen_val_batches", 4))
        self.shuffle_prefix_diag = bool(gen_cfg.get("shuffle_prefix_diag", False))


    def prepare_trainer(self):
        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)

    @nvtx.annotate("Training Section", color="green")
    def train(self, prof = False, load=False):
        """Entrena el modelo Imitator.
        returns:
            train_loss: float, loss de entrenamiento
            val_loss: float, loss de validación
        """
        print("LR:", self.learning_rate)
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            foreach=True
        )
        
        def linear_warmup_cosine_decay(current_step, warmup_steps, total_steps):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + torch.cos(
                torch.tensor((current_step - warmup_steps) / (total_steps - warmup_steps) * 3.1415926535))
            ).item()

        warmup_steps = 5 * len(self.train_loader)  # p.ej. 5 epochs de warm-up
        total_steps = self.epochs * len(self.train_loader)

        lr_lambda = lambda step: linear_warmup_cosine_decay(step, warmup_steps, total_steps)

        # log_temp must be added BEFORE LambdaLR so the scheduler sees both
        # param groups from the start (strict=True zip in scheduler.step() fails
        # if the number of groups grows after the scheduler is created).
        if self.loss_type == "ce_vocab" and self.log_temp is not None:
            self.optimizer.add_param_group({
                "params": [self.log_temp],
                "lr": self.learning_rate,
                "weight_decay": 0.0,
            })

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        if self.load_previous_model:
            self.ckpt_mgr.load_checkpoint(self.model, self.optimizer, self.scheduler)

        self.prepare_trainer()
        self.prof = prof

        if self.loss_type == "ce_vocab":
            self._log_unigram_baseline()

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch)
            val_loss = self._val(epoch)

            if epoch == 1:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            elif epoch == self.epochs - 1:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            elif (epoch % self.checkpoint_interval == 0 and epoch != 0) :
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)
            elif self.early_stopping.stop:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler)

            if self.early_stopping.improved:
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler, tag="best")

            if self.scheduler is not None:
                self.scheduler.step()

            if self.early_stopping.stop:
                break

        return train_loss, val_loss

    @nvtx.annotate("Distributed Training Section", color="green")
    def train_dist(self, rank, dist, stub):
        from src.mslm.distributed import data_pb2, data_pb2_grpc
        import io

        """Entrena el modelo Imitator distribuido.
        returns:
            train_loss: float, loss de entrenamiento
            val_loss: float, loss de validación
        """
        self.distributed = dist
        def save_model_dist():
                buf = io.BytesIO()
                torch.save(self.model, buf)
                req = data_pb2.SaveModelRequest(
                    model_bytes=buf.getvalue(),
                    model_name = f"{epoch}"
                )
                resp = stub.SaveModel(req)
                print("=== SAVE MODEL ===")
                print(" success:", resp.success)
                print(" message:", resp.message)

        print("LR:", self.learning_rate)
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-4,
            foreach=True
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
        self.prepare_trainer()

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch)
            if rank == 0:
                if epoch == 1:
                    save_model_dist()
                elif (epoch % self.checkpoint_interval == 0 and epoch != 0) or (epoch == self.epochs - 1):
                    save_model_dist()

            val_loss = self.accelerator.gather(torch.tensor(val_loss, device=self.device.type)).mean().item()
            self.scheduler.step(val_loss)
            if self.early_stopping.stop and rank == 0:
                save_model_dist()

            if epoch % self.log_interval == 0:
                tqdm.write(f"\nEpoch: {epoch}.\t Total loss: {train_loss/len(self.train_loader)}")

        return train_loss, val_loss

    @nvtx.annotate("Train: Train Epoch", color="green")
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        mse_loss = 0
        cossim_loss = 0
        top5_acc = 0.0
        self._sigreg_sum = 0.0
        self._sigreg_n = 0
        self._infonce_sum = 0.0
        self._infonce_n = 0
        self._grad_norm_sum = 0.0
        self._grad_norm_n = 0
        for batch in self.train_loader:
            keypoint, frames_padding_mask, embedding, mask_embedding = batch[:4]
            token_ids = batch[4] if len(batch) > 4 else None
            if self.save_tb_model and epoch == 1 and not getattr(self, "graph_added", False):
                print("Saving graph")
                self.writer.add_graph(self.model, (keypoint, frames_padding_mask))
                self.graph_added = True

            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad(set_to_none=True)
                train_loss, mse, cossim, top5 = self._train_batch(keypoint, frames_padding_mask, embedding, mask_embedding, token_ids)

            if self.distributed is not None:
                loss_tensor = loss.to(self.device)
                self.distributed.all_reduce(loss_tensor, op=self.distributed.ReduceOp.SUM)
                loss = (loss_tensor) / self.distributed.get_world_size()
                if self.distributed.get_rank() == 0:
                    print(f"World-avg train loss: {loss:.4f}")
            else:
                total_loss += train_loss
                mse_loss += mse
                cossim_loss += cossim
                top5_acc += top5

        final_train_loss = total_loss.item()/len(self.train_loader)
        final_train_loss_mse = mse_loss.item()/len(self.train_loader)
        final_train_loss_cossim = cossim_loss.item()/len(self.train_loader)
        final_top5_train = top5_acc if isinstance(top5_acc, float) else top5_acc.item()
        final_top5_train /= len(self.train_loader)

        self.writer.add_scalar("Loss/train", final_train_loss, epoch)
        if self.loss_type == "ce_vocab":
            self.writer.add_scalar("Loss/train_ce", final_train_loss_mse, epoch)
            self.writer.add_scalar("Metrics/train_token_acc", final_train_loss_cossim, epoch)
            self.writer.add_scalar("Metrics/train_token_acc_top5", final_top5_train, epoch)
            if self.log_temp is not None:
                self.writer.add_scalar("Params/log_temp", self.log_temp.item(), epoch)
                self.writer.add_scalar("Params/temp", self.log_temp.exp().item(), epoch)
        elif self.loss_type == "ce_ar":
            self.writer.add_scalar("Loss/train_ce_ar", final_train_loss_mse, epoch)
            self.writer.add_scalar("Metrics/train_token_acc", final_train_loss_cossim, epoch)
            self.writer.add_scalar("Metrics/train_token_acc_top5", final_top5_train, epoch)
        else:
            self.writer.add_scalar("Loss/train_mse", final_train_loss_mse, epoch)
            self.writer.add_scalar("Loss/train_cosim", final_train_loss_cossim, epoch)

        sigreg_str = ""
        if self.sigreg_enabled and self._sigreg_n:
            final_sigreg = self._sigreg_sum / self._sigreg_n
            self.writer.add_scalar("Loss/train_sigreg", final_sigreg, epoch)
            sigreg_str = f" SIGReg: {final_sigreg:.4f} (λ={self.sigreg_lambda})"

        if self.infonce_enabled and self._infonce_n:
            self.writer.add_scalar("Loss/train_infonce", self._infonce_sum / self._infonce_n, epoch)

        if self._grad_norm_n:
            self.writer.add_scalar("Train/grad_norm_imitator", self._grad_norm_sum / self._grad_norm_n, epoch)

        if epoch % self.log_interval == 0:
            if self.loss_type in ("ce_vocab", "ce_ar"):
                tqdm.write(f"\nEpoch: {epoch}.\n Train loss: {final_train_loss} CE: {final_train_loss_mse} TokenAcc: {final_train_loss_cossim} Top5: {final_top5_train:.4f}{sigreg_str}")
            else:
                tqdm.write(f"\nEpoch: {epoch}.\n Train loss: {final_train_loss} MSE: {final_train_loss_mse} Cossim: {final_train_loss_cossim}{sigreg_str}")

        return total_loss

    def _forward_loss(self, keypoint, frames_padding_mask, embedding, mask_embedding, token_ids=None):
        with self.accelerator.autocast():
            output, _ = self.model(keypoint, frames_padding_mask)
            result = self.criterion(output, embedding, mask_embedding, token_ids)
            if len(result) == 4:
                loss, mse, cossim, top5 = result
            else:
                loss, mse, cossim = result
                top5 = torch.zeros((), device=loss.device)
            if self.sigreg_enabled:
                sr = sigreg_loss(output, None, **self.sigreg_kwargs)
                loss = loss + self.sigreg_lambda * sr
                self._sigreg_sum += float(sr.detach())
                self._sigreg_n += 1
        return loss, mse, cossim, top5

    @nvtx.annotate("Train: Train Batch", color="green")
    def _train_batch(self, keypoint, frames_padding_mask, embedding, mask_embedding, token_ids=None):
        batch_loss = 0.0
        batch_mse, batch_cossim, batch_top5 = 0.0, 0.0, 0.0

        batch_size = keypoint.size(0)
        start = 0
        end = keypoint.size(0)
        if self.batch_sampling:
            n_sub_batch = (batch_size + self.sub_batch - 1) // self.sub_batch

        with nvtx.annotate("Sub_batch", color="blue"):
            with torch.autograd.set_detect_anomaly(True):
                for i in range(n_sub_batch):
                    if self.batch_sampling:
                        start = i * self.sub_batch
                        end = min(start + self.sub_batch, batch_size)
                        with nvtx.annotate("Forward Pass", color="blue"):
                            loss, mse, cossim, top5 = self._forward_loss(keypoint[start:end],
                                                        frames_padding_mask[start:end],
                                                        embedding[start:end],
                                                        mask_embedding[start:end],
                                                        token_ids[start:end] if token_ids is not None else None)
                        if self.batch_sampling:
                            # OUT-OF-PLACE a propósito: imitator_ce_loss devuelve (ce, ce.detach(), …),
                            # así que `loss` y `mse` COMPARTEN storage. Un `/=` in-place dividía la CE
                            # dos veces (loss y mse), dejando el backward sobre ce/n_sub_batch² →
                            # gradiente deflactado por n_sub_batch (lr efectivo /16) y métricas corruptas.
                            loss = loss / n_sub_batch
                            mse = mse / n_sub_batch
                            cossim = cossim / n_sub_batch
                            top5 = top5 / n_sub_batch

                        with nvtx.annotate("Backward Pass", color="blue"):
                            torch.autograd.set_detect_anomaly(True)
                            self.accelerator.backward(loss)
                        batch_loss += loss.detach()
                        batch_mse += mse.detach()
                        batch_cossim += cossim.detach()
                        batch_top5 += top5.detach()

            with nvtx.annotate("Step", color="blue"):
                with torch.no_grad():
                    norm_sq = sum(
                        p.grad.detach().norm() ** 2
                        for p in self.model.parameters()
                        if p.grad is not None
                    )
                    self._grad_norm_sum += float(norm_sq.sqrt())
                    self._grad_norm_n += 1
                params_to_clip = list(self.model.parameters())
                if self.loss_type == "ce_vocab" and self.log_temp is not None:
                    params_to_clip.append(self.log_temp)
                self.accelerator.clip_grad_norm_(params_to_clip, max_norm=self.grad_clip)
                self.optimizer.step()

        return batch_loss, batch_mse, batch_cossim, batch_top5

    @nvtx.annotate("Validation Section", color="green")
    def _val(self, epoch):
        self.model.eval()
        val_loss = 0
        mse_loss = 0
        cossim_loss = 0
        top5_acc = 0.0
        for batch in self.val_loader:
            keypoint, frames_padding_mask, embedding, mask_embedding = batch[:4]
            token_ids = batch[4] if len(batch) > 4 else None
            loss, mse, cossim, top5 = self._val_batch(keypoint, frames_padding_mask, embedding, mask_embedding, token_ids)
            if self.distributed is not None:
                loss_tensor = loss.to(self.device)
                self.distributed.all_reduce(loss_tensor, op=self.distributed.ReduceOp.SUM)
                val_loss = (loss_tensor) / self.distributed.get_world_size()
                if self.distributed.get_rank() == 0:
                    print(f"World-avg val loss: {loss:.4f}")
            else:
                val_loss += loss
                mse_loss += mse
                cossim_loss += cossim
                top5_acc += top5

        final_val_loss = val_loss.item() / len(self.val_loader)
        final_mse_loss = mse_loss.item() / len(self.val_loader)
        final_cossim_loss = cossim_loss.item() / len(self.val_loader)
        final_top5_val = top5_acc if isinstance(top5_acc, float) else top5_acc.item()
        final_top5_val /= len(self.val_loader)

        self.writer.add_scalar("Loss/val", final_val_loss, epoch)
        if self.loss_type == "ce_vocab":
            self.writer.add_scalar("Loss/val_ce", final_mse_loss, epoch)
            self.writer.add_scalar("Metrics/val_token_acc", final_cossim_loss, epoch)
            self.writer.add_scalar("Metrics/val_token_acc_top5", final_top5_val, epoch)
            # Guardarraíl anti-artefacto: la CE con label smoothing no puede bajar del piso
            # ε*log(V). Una val_ce por debajo delata una métrica corrupta (cf. v115.1).
            floor = ce_floor(self.embed_table.shape[0], self.label_smoothing)
            self.writer.add_scalar("Loss/val_ce_floor", floor, epoch)
            if self.label_smoothing > 0 and final_mse_loss < floor - 1e-3:
                tqdm.write(f"⚠️  val_ce={final_mse_loss:.4f} < piso teórico {floor:.4f} "
                           f"(ε={self.label_smoothing}, V={self.embed_table.shape[0]}). "
                           f"Métrica CORRUPTA — no usar para selección de modelo.")
        elif self.loss_type == "ce_ar":
            self.writer.add_scalar("Loss/val_ce_ar", final_mse_loss, epoch)
            self.writer.add_scalar("Metrics/val_token_acc", final_cossim_loss, epoch)
            self.writer.add_scalar("Metrics/val_token_acc_top5", final_top5_val, epoch)
        else:
            self.writer.add_scalar("Loss/val_mse", final_mse_loss, epoch)
            self.writer.add_scalar("Loss/val_cossim", final_cossim_loss, epoch)

        if epoch % self.log_interval == 0:
            if self.loss_type in ("ce_vocab", "ce_ar"):
                tqdm.write(f"Validation loss: {final_val_loss} CE: {final_mse_loss} TokenAcc: {final_cossim_loss} Top5: {final_top5_val:.4f}")
            else:
                tqdm.write(f"Validation loss: {final_val_loss} MSE: {final_mse_loss} Cossim: {final_cossim_loss}")

        if self.diagnostics_enabled and (epoch % self.diag_interval == 0):
            self._run_diagnostics(epoch)

        if (self.loss_type == "ce_ar" and self.bridge is not None
                and epoch % self.gen_interval == 0):
            self._run_generation_eval(epoch)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        self.early_stopping(final_val_loss, epoch=epoch)
        return final_val_loss

    @torch.no_grad()
    def _log_unigram_baseline(self):
        V = self.embed_table.shape[0]
        counts = torch.zeros(V, dtype=torch.long)
        for batch in self.train_loader:
            ids = batch[4] if len(batch) > 4 else None
            if ids is None:
                return
            flat = ids.reshape(-1).cpu()
            valid = flat[flat != -100]
            counts += torch.bincount(valid, minlength=V)
        total = counts.sum().item()
        if total == 0:
            return
        probs = counts.float() / total
        nz = probs[probs > 0]
        uni_ce   = -(nz * nz.log()).sum().item()
        uni_acc1 = probs.max().item()
        uni_acc5 = probs.topk(5).values.sum().item()
        self.writer.add_scalar("Baseline/unigram_ce",       uni_ce,   0)
        self.writer.add_scalar("Baseline/unigram_acc_top1", uni_acc1, 0)
        self.writer.add_scalar("Baseline/unigram_acc_top5", uni_acc5, 0)
        tqdm.write(f"[Unigram baseline] CE={uni_ce:.3f} acc1={uni_acc1:.4f} acc5={uni_acc5:.4f}")

    @nvtx.annotate("Diagnostics: Collapse", color="orange")
    @torch.no_grad()
    def _run_diagnostics(self, epoch):
        """Calcula y registra métricas de colapso de embeddings sobre el set de validación.

        Acumula los tokens válidos (no padding) de los embeddings PREDICHOS y, opcionalmente,
        de los OBJETIVO sobre las primeras ``diag_n_batches`` batches; luego calcula el rango
        efectivo, el coseno medio entre pares, el colapso por dimensión, etc.
        """
        self.model.eval()
        pred_chunks, target_chunks = [], []
        sb = self.sub_batch if self.batch_sampling else None

        for i, batch in enumerate(self.val_loader):
            keypoint, frames_padding_mask, embedding, mask_embedding = batch[:4]
            if i >= self.diag_n_batches:
                break
            B = keypoint.size(0)
            step = sb or B
            # Sub-batchear el forward (igual que el entrenamiento) para no saturar la GPU.
            for s in range(0, B, step):
                e = min(s + step, B)
                with self.accelerator.autocast():
                    output, _ = self.model(keypoint[s:e], frames_padding_mask[s:e])
                # Alinear longitudes igual que la pérdida y quedarse con tokens válidos.
                L = min(output.size(1), embedding.size(1))
                valid = ~mask_embedding[s:e, :L]                # (b, L) True = válido
                pred_chunks.append(output[:, :L][valid].float().cpu())
                if self.diag_compare_target:
                    target_chunks.append(embedding[s:e, :L][valid].float().cpu())
                del output
        torch.cuda.empty_cache()

        if not pred_chunks:
            return

        pred = torch.cat(pred_chunks, dim=0)                    # (M, D)
        target = torch.cat(target_chunks, dim=0) if target_chunks else None

        metrics = compute_collapse_metrics(
            pred, target_embs=target, per_dim_std_eps=self.diag_eps
        )

        if self.accelerator.is_main_process:
            for name, value in metrics.items():
                self.writer.add_scalar(f"Collapse/{name}", value, epoch)
            ratio = metrics.get("effrank_ratio_vs_target")
            alert = ""
            if self.diag_alert_ratio > 0 and ratio is not None and ratio < self.diag_alert_ratio:
                alert = "  ⚠️ POSIBLE COLAPSO"
            tqdm.write(
                f"[Collapse] epoch {epoch}: "
                f"effrank={metrics['effective_rank']:.1f} "
                f"ratio={ratio if ratio is None else round(ratio, 3)} "
                f"cos={metrics['pairwise_cosine_mean']:.3f} "
                f"dim_collapsed={metrics['per_dim_collapsed_frac']:.2f}{alert}"
            )

    @torch.no_grad()
    def _run_generation_eval(self, epoch: int):
        """Greedy-decode from soft prefix over a val subset and log chrF.

        Runs every ``gen_interval`` epochs.  Prints up to 5 example pairs
        (reference vs hypothesis) for manual inspection.

        Imitator forward is sub-batched to stay within GPU memory; then all
        prefix vectors are stacked and a single batched generate() call is made
        so that the 50 greedy steps run in parallel across the full batch.
        """
        self.model.eval()
        hyps: "list[str]" = []
        refs: "list[str]" = []
        examples_shown = False
        pad_id = getattr(self.bridge.tokenizer, "pad_token_id", None) or 0

        step = self.sub_batch if self.batch_sampling else None

        for i, batch in enumerate(self.val_loader):
            if i >= self.gen_val_batches:
                break
            keypoint = batch[0]
            frames_padding_mask = batch[1]
            token_ids = batch[4] if len(batch) > 4 else None
            if token_ids is None:
                continue

            B = keypoint.size(0)
            sub = step or B
            prefix_chunks = []
            for s in range(0, B, sub):
                e = min(s + sub, B)
                with self.accelerator.autocast():
                    prefix_chunk, _ = self.model(keypoint[s:e], frames_padding_mask[s:e])
                prefix_chunks.append(prefix_chunk.float())  # collect in float32

            # Stack all prefixes and generate in one batched call (no sub-batch).
            # @torch.no_grad() + no gradient storage keeps memory low enough.
            prefix = torch.cat(prefix_chunks, dim=0)  # [B, K, H]
            gen_ids = self.bridge.generate(prefix, max_new_tokens=self.gen_max_tokens)
            gen_texts = self.bridge.decode(gen_ids)

            ref_ids = token_ids.clone()
            ref_ids[ref_ids == -100] = pad_id
            ref_texts = self.bridge.decode(ref_ids)

            hyps.extend(gen_texts)
            refs.extend(ref_texts)

            if not examples_shown and self.accelerator.is_main_process:
                tqdm.write(f"\n[Gen ep{epoch}] Ejemplos generados:")
                for j in range(min(5, len(gen_texts))):
                    tqdm.write(f"  REF: {ref_texts[j][:120]}")
                    tqdm.write(f"  HYP: {gen_texts[j][:120]}")
                    tqdm.write("")
                examples_shown = True

        if not hyps:
            return

        chrf = _compute_chrf(hyps, refs)
        if self.accelerator.is_main_process:
            self.writer.add_scalar("Gen/val_chrf", chrf, epoch)
            tqdm.write(f"[Gen ep{epoch}] chrF={chrf:.2f} ({len(hyps)} muestras)")
            if chrf > self.best_chrf:
                self.best_chrf = chrf
                self.ckpt_mgr.save_checkpoint(self.model, epoch, self.optimizer, self.scheduler, tag="best_chrf")
                tqdm.write(f"  ↑ best chrF: {chrf:.2f} (ep {epoch})")

        if getattr(self, "shuffle_prefix_diag", False):
            hyps_shuf: "list[str]" = []
            refs_shuf: "list[str]" = []
            for i, batch in enumerate(self.val_loader):
                if i >= self.gen_val_batches:
                    break
                keypoint = batch[0]
                frames_padding_mask = batch[1]
                token_ids = batch[4] if len(batch) > 4 else None
                if token_ids is None:
                    continue
                B = keypoint.size(0)
                sub = self.sub_batch if self.batch_sampling else B
                prefix_chunks = []
                for s in range(0, B, sub):
                    e = min(s + sub, B)
                    with self.accelerator.autocast():
                        prefix_chunk, _ = self.model(keypoint[s:e], frames_padding_mask[s:e])
                    prefix_chunks.append(prefix_chunk.float())
                prefix_full = torch.cat(prefix_chunks, dim=0)
                prefix_shuffled = torch.roll(prefix_full, 1, dims=0)
                gen_ids_shuf = self.bridge.generate(prefix_shuffled, max_new_tokens=self.gen_max_tokens)
                gen_texts_shuf = self.bridge.decode(gen_ids_shuf)
                ref_ids = token_ids.clone()
                pad_id_shuf = getattr(self.bridge.tokenizer, "pad_token_id", None) or 0
                ref_ids[ref_ids == -100] = pad_id_shuf
                ref_texts_shuf = self.bridge.decode(ref_ids)
                hyps_shuf.extend(gen_texts_shuf)
                refs_shuf.extend(ref_texts_shuf)
            if hyps_shuf:
                chrf_shuf = _compute_chrf(hyps_shuf, refs_shuf)
                if self.accelerator.is_main_process:
                    self.writer.add_scalar("Gen/val_chrf_shuffled", chrf_shuf, epoch)
                    tqdm.write(f"  chrF (shuffled prefix): {chrf_shuf:.2f}  gap: {chrf - chrf_shuf:+.2f}")

    @nvtx.annotate("Val: Validate Batch", color="green")
    @torch.no_grad()
    def _val_batch(self, keypoint, frames_padding_mask, embedding, mask_embedding, token_ids=None) -> t.Tuple[float, float, float, float]:
        batch_loss = 0.0
        batch_mse = 0.0
        batch_cossim = 0.0
        batch_top5 = 0.0

        batch_size = keypoint.size(0)
        start = 0
        end = keypoint.size(0)

        if self.batch_sampling:
            n_sub_batch = (batch_size + self.sub_batch - 1) // self.sub_batch

        with nvtx.annotate("Val: Forward + Loss", color="blue"):
            for i in range(n_sub_batch):
                if self.batch_sampling:
                    start = i * self.sub_batch
                    end = min(start + self.sub_batch, batch_size)
                with nvtx.annotate("Forward Pass", color="blue"):
                    loss, mse, cossim, top5 = self._forward_loss(keypoint[start:end],
                                                frames_padding_mask[start:end],
                                                embedding[start:end],
                                                mask_embedding[start:end],
                                                token_ids[start:end] if token_ids is not None else None)
                if self.batch_sampling:
                    # OUT-OF-PLACE: ver nota en _train_batch. `loss` (=ce) y `mse` (=ce.detach())
                    # comparten storage; un `/=` in-place deflactaba la CE por n_sub_batch².
                    loss = loss / n_sub_batch
                    mse = mse / n_sub_batch
                    cossim = cossim / n_sub_batch
                    top5 = top5 / n_sub_batch

                batch_loss += loss.detach()
                batch_mse += mse.detach()
                batch_cossim += cossim.detach()
                batch_top5 += top5.detach()

        return batch_loss, batch_mse, batch_cossim, batch_top5
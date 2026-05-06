import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class PointGradCAM:
    """
    Grad-CAM for point-cloud models (OpenPCDet / PointRCNN).

    - Hooks a named module (target_layer_name)
    - Captures the best activation tensor from module outputs (prefers feature maps, not xyz)
    - Captures gradients via tensor-level hook + acts.retain_grad()

    IMPORTANT:
    - target_fn MUST return a differentiable scalar connected to the activation.
    - Avoid post-NMS pred_scores as backprop target; prefer pre-NMS logits (e.g., ROI head logits).
    """

    def __init__(self, model: torch.nn.Module, target_layer_name: str, device=None):
        self.model = model
        self.target_layer_name = target_layer_name
        self.device = device or next(model.parameters()).device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self.handles = []
        self._act_tensor_hook = None
        self.verbose = False

        self._register_hooks()

    def _find_module(self, name: str) -> torch.nn.Module:
        for n, m in self.model.named_modules():
            if n == name:
                return m
        available = [n for n, _ in self.model.named_modules()]
        raise ValueError(
            f"Module named '{name}' not found in model.\n"
            f"Tip: run your script with --list-modules.\n"
            f"Available modules (count={len(available)}): {available[:200]}{' ...' if len(available)>200 else ''}"
        )

    @staticmethod
    def _collect_tensors(obj: Any, out: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        if out is None:
            out = []
        if isinstance(obj, torch.Tensor):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                PointGradCAM._collect_tensors(o, out)
        elif isinstance(obj, dict):
            for v in obj.values():
                PointGradCAM._collect_tensors(v, out)
        return out

    @staticmethod
    def _pick_activation(tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
        cands = [t for t in tensors if t.is_floating_point()]
        if not cands:
            return None

        def score(t: torch.Tensor) -> Tuple[int, int, int, int, int]:
            req = 1 if t.requires_grad else 0
            dim3 = 1 if t.dim() == 3 else 0
            dim2 = 1 if t.dim() == 2 else 0
            ch = 0
            if t.dim() == 3:
                ch = max(int(t.shape[1]), int(t.shape[2]))
            xyz_penalty = 1 if (t.dim() == 3 and (t.shape[-1] == 3 or t.shape[1] == 3)) else 0
            return (req, dim3, dim2, ch, t.numel() - 10_000_000 * xyz_penalty)

        return sorted(cands, key=score, reverse=True)[0]

    def _register_hooks(self) -> None:
        module = self._find_module(self.target_layer_name)

        def forward_hook(m, inp, out):
            if self.verbose:
                print(f"[GradCAM] forward_hook '{self.target_layer_name}' ({m.__class__.__name__})")

            tensors = self._collect_tensors(out)
            if not tensors:
                tensors = self._collect_tensors(inp)

            act = self._pick_activation(tensors)
            self.activations = act

            if self.verbose:
                print(
                    f"[GradCAM] picked activation: "
                    f"shape={getattr(act, 'shape', None)}, "
                    f"dtype={getattr(act, 'dtype', None)}, "
                    f"requires_grad={getattr(act, 'requires_grad', None)}"
                )

            self.gradients = None
            if isinstance(act, torch.Tensor):
                try:
                    act.retain_grad()
                except Exception:
                    if self.verbose:
                        print("[GradCAM] retain_grad() failed (ok for leaf tensors)")

                if self._act_tensor_hook is not None:
                    try:
                        self._act_tensor_hook.remove()
                    except Exception:
                        pass
                    self._act_tensor_hook = None

                def _capture_grad(g):
                    self.gradients = g
                    if self.verbose:
                        print(f"[GradCAM] activation grad hook fired; grad shape={getattr(g,'shape',None)}")

                try:
                    self._act_tensor_hook = act.register_hook(_capture_grad)
                except Exception:
                    if self.verbose:
                        print("[GradCAM] WARNING: failed to register tensor grad hook on activation")

        self.handles.append(module.register_forward_hook(forward_hook))

    def remove_hooks(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []
        if self._act_tensor_hook is not None:
            try:
                self._act_tensor_hook.remove()
            except Exception:
                pass
            self._act_tensor_hook = None

    @staticmethod
    def _normalize_np(cam: np.ndarray) -> np.ndarray:
        if cam.size == 0:
            return cam
        mn = float(cam.min())
        mx = float(cam.max())
        if mx - mn > 1e-8:
            return (cam - mn) / (mx - mn)
        return cam - mn

    @staticmethod
    def _to_cam_1d(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
        if acts.dim() != grads.dim():
            raise RuntimeError(f"acts/grads dim mismatch: acts={acts.shape}, grads={grads.shape}")

        if acts.dim() == 4:
            if acts.shape[-1] == 1:
                acts = acts.squeeze(-1)
                grads = grads.squeeze(-1)
            elif acts.shape[-2] == 1:
                acts = acts.squeeze(-2)
                grads = grads.squeeze(-2)
            else:
                B, C, H, W = acts.shape
                acts = acts.view(B, C, H * W)
                grads = grads.view(B, C, H * W)

        if acts.dim() != 3 or grads.dim() != 3:
            raise RuntimeError(f"Expected 3D acts/grads after normalize, got acts={acts.shape}, grads={grads.shape}")

        if acts.shape[-1] == 3 or acts.shape[1] == 3:
            raise RuntimeError(
                f"Activation looks xyz-like (shape={tuple(acts.shape)}). "
                f"Your hook likely captured coordinates, not features."
            )

        if acts.shape[1] == grads.shape[1]:
            weights = grads.mean(dim=2, keepdim=True)
            cam = (weights * acts).sum(dim=1)
            return cam

        if acts.shape[2] == grads.shape[2]:
            weights = grads.mean(dim=1, keepdim=True)
            cam = (weights * acts).sum(dim=2)
            return cam

        B = acts.shape[0]
        acts_flat = acts.reshape(B, acts.shape[1], -1)
        grads_flat = grads.reshape(B, grads.shape[1], -1)
        weights = grads_flat.mean(dim=2, keepdim=True)
        cam = (weights * acts_flat).sum(dim=1)
        return cam

    def attribute(
        self,
        input_dict: Dict[str, Any],
        target_fn,
        retain_graph: bool = False,
        return_outputs: bool = False
    ):
        self.model.zero_grad(set_to_none=True)

        training_state = self.model.training
        self.model.eval()

        try:
            outputs = self.model(input_dict)

            acts = self.activations
            if acts is None:
                raise RuntimeError(
                    f"Layer '{self.target_layer_name}' did not run or no activation selected. "
                    f"Check the module name and that it is executed in forward."
                )

            target = target_fn(outputs)
            if not isinstance(target, torch.Tensor):
                raise ValueError("target_fn must return a torch.Tensor")

            target_scalar = target.sum() if target.dim() > 0 else target

            if self.verbose:
                print(
                    f"[GradCAM] target_scalar: shape={getattr(target_scalar,'shape',None)}, "
                    f"requires_grad={getattr(target_scalar,'requires_grad',None)}, "
                    f"grad_fn={getattr(target_scalar,'grad_fn',None)}"
                )

            if not bool(getattr(target_scalar, "requires_grad", False)):
                raise RuntimeError(
                    "target_fn returned a non-differentiable tensor (requires_grad=False). "
                    "Usually caused by using post-processed outputs (after NMS), e.g. pred_scores."
                )

            target_scalar.backward(retain_graph=retain_graph)

            grads = self.gradients
            if grads is None and isinstance(acts, torch.Tensor):
                grads = acts.grad
            if grads is None:
                raise RuntimeError("No gradients captured for activation.")

            acts_cpu = acts.detach().cpu()
            grads_cpu = grads.detach().cpu()

            cam = self._to_cam_1d(acts_cpu, grads_cpu)
            cam = F.relu(cam)

            cam_list: List[np.ndarray] = []
            for b in range(cam.shape[0]):
                arr = cam[b].numpy()
                cam_list.append(self._normalize_np(arr))

            if return_outputs:
                return cam_list, outputs
            return cam_list

        finally:
            if training_state:
                self.model.train()

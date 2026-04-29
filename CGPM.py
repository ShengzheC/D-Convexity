import torch
from torch import nn
from loss import C2nd_loss, C1st_loss, fidelity_loss

class UnrolledCGPM(nn.Module):
    def __init__(
        self,
        steps=100,
        lr=1e-2,
        method="c2",
        beta1=0.9,
        beta2=0.999,
        adam_eps=1e-8,
        backprop_to_backbone=False,
        c1_window_size=7,
        c2_margin=1e-3,
        c2_softplus_beta=1.0,
        c2_eps=1e-8,
        c2_grad_threshold=1e-5,
    ):
        super().__init__()
        self.steps = steps
        self.lr = lr
        self.method = method
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_eps = adam_eps
        self.backprop_to_backbone = backprop_to_backbone

        self.c1_window = c1_window_size

        self.c2_kwargs = dict(
            margin=c2_margin,
            softplus_beta=c2_softplus_beta,
            eps=c2_eps,
            grad_threshold=c2_grad_threshold,
        )

    def convexity_loss(self, u):
        if self.method == "c2":
            return C2nd_loss(u, **self.c2_kwargs)
        elif self.method == "c1":
            return C1st_loss(u, self.c1_window)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def forward(self, logits):
        logits_norm = logits / (logits.abs().amax(dim=(2, 3), keepdim=True) + 1e-8)
        u0 = torch.sigmoid(logits_norm).detach()

        with torch.enable_grad():
            if self.backprop_to_backbone:
                o = logits_norm.clone().requires_grad_(True)
            else:
                o = logits_norm.clone().detach().requires_grad_(True)

            m = torch.zeros_like(o)
            v = torch.zeros_like(o)

            history = []
            for t in range(1, self.steps + 1):
                u = torch.sigmoid(o)
                loss_fid = fidelity_loss(u, u0)
                loss_convex = self.convexity_loss(u)
                loss = loss_fid + loss_convex

                # Autograd with Adam gradient descent.
                grad_o = torch.autograd.grad(
                    loss,
                    o,
                    create_graph=self.backprop_to_backbone,
                    retain_graph=self.backprop_to_backbone,
                    only_inputs=True,
                )[0]

                m = self.beta1 * m + (1.0 - self.beta1) * grad_o
                v = self.beta2 * v + (1.0 - self.beta2) * (grad_o * grad_o)

                m_hat = m / (1.0 - self.beta1 ** t)
                v_hat = v / (1.0 - self.beta2 ** t)

                o = o - self.lr * m_hat / (torch.sqrt(v_hat) + self.adam_eps)

                history.append({
                    "loss_fid": loss_fid.detach(),
                    "loss_convex": loss_convex.detach(),
                })

        refined = torch.sigmoid(o)
        return refined, history

class SegModelWithCGPM(nn.Module):
    def __init__(self, backbone, refine_steps=100, refine_lr=1e-2, backprop_to_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.refiner = UnrolledCGPM(
            steps=refine_steps,
            lr=refine_lr,
            method="c2",
            backprop_to_backbone=backprop_to_backbone,
        )

    def forward(self, x):
        logits = self.backbone(x)
        refined, _ = self.refiner(logits)
        return (refined >= 0.5).to(dtype=torch.uint8)
    
# example use:
# We recommend using CGPM with 2nd loss as plug-and-play module at inference. 
# Although incorportating it in training is possible, we found no obvious advantages in doing that.

# model = UNet2D().to(device)
# model.load_state_dict(ckpt)
# model.eval()
# SegCGPM = SegModelWithCGPM(model, backprop_to_backbone=False)
# predicted_mask = SegCGPM(images)
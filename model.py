import torch
import torch.nn as nn
from .networks import PatchGAN, UnetGenerator

class Pix2Pix(nn.Module):
    """Create a Pix2Pix class. It is a model for image to image translation tasks.
    By default, the model uses a Unet architecture for generator with transposed
    convolution. The discriminator is 70x70 PatchGAN discriminator, by default.
    """

    def __init__(
        self,
        c_in: int = 3,
        c_out: int = 3,
        is_train: bool = True,
        netD: str = "patch",
        lambda_L1: float = 100.0,
        is_CGAN: bool = True,
        use_upsampling: bool = False,
        mode: str = "nearest",
        c_hid: int = 64,
        n_layers: int = 3,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
    ):
        super(Pix2Pix, self).__init__()
        self.is_CGAN = is_CGAN
        self.lambda_L1 = lambda_L1
        self.is_train = is_train

        self.gen = UnetGenerator(
            c_in=c_in, c_out=c_out, use_upsampling=use_upsampling, mode=mode
        )
        self.gen = self.gen.apply(self.weights_init)

        if self.is_train:
            disc_in = c_in + c_out if is_CGAN else c_out
            self.disc = PatchGAN(
                c_in=disc_in, c_hid=c_hid, mode=netD, n_layers=n_layers
            )
            self.disc = self.disc.apply(self.weights_init)
            self.gen_optimizer = torch.optim.Adam(
                self.gen.parameters(), lr=lr, betas=(beta1, beta2)
            )
            self.disc_optimizer = torch.optim.Adam(
                self.disc.parameters(), lr=lr, betas=(beta1, beta2)
            )
            self.criterion = nn.BCEWithLogitsLoss()
            self.criterion_L1 = nn.L1Loss()

    def forward(self, x: torch.Tensor):
        return self.gen(x)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def _get_disc_inputs(self, real_images: torch.Tensor, target_images: torch.Tensor, fake_images: torch.Tensor):
        if self.is_CGAN:
            real_AB = torch.cat([real_images, target_images], dim=1)
            fake_AB = torch.cat([real_images, fake_images.detach()], dim=1)
        else:
            real_AB = target_images
            fake_AB = fake_images.detach()
        return real_AB, fake_AB

    def _get_gen_inputs(self, real_images: torch.Tensor, fake_images: torch.Tensor):
        if self.is_CGAN:
            fake_AB = torch.cat([real_images, fake_images], dim=1)
        else:
            fake_AB = fake_images
        return fake_AB

    def step_discriminator(self, real_images: torch.Tensor, target_images: torch.Tensor, fake_images: torch.Tensor):
        real_AB, fake_AB = self._get_disc_inputs(real_images, target_images, fake_images)
        pred_real = self.disc(real_AB)
        pred_fake = self.disc(fake_AB)
        lossD_real = self.criterion(pred_real, torch.ones_like(pred_real))
        lossD_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake))
        lossD = (lossD_real + lossD_fake) * 0.5
        return lossD

    def step_generator(self, real_images: torch.Tensor, target_images: torch.Tensor, fake_images: torch.Tensor):
        fake_AB = self._get_gen_inputs(real_images, fake_images)
        pred_fake = self.disc(fake_AB)
        lossG_GaN = self.criterion(pred_fake, torch.ones_like(pred_fake))
        lossG_L1 = self.criterion_L1(fake_images, target_images)
        lossG = lossG_GaN + self.lambda_L1 * lossG_L1
        return lossG, {"loss_G": lossG.item(), "loss_G_GAN": lossG_GaN.item(), "loss_G_L1": lossG_L1.item()}

    def train_step(self, real_images: torch.Tensor, target_images: torch.Tensor):
        fake_images = self.forward(real_images)
        self.disc_optimizer.zero_grad()
        lossD = self.step_discriminator(real_images, target_images, fake_images)
        lossD.backward()
        self.disc_optimizer.step()
        self.gen_optimizer.zero_grad()
        lossG, G_losses = self.step_generator(real_images, target_images, fake_images)
        lossG.backward()
        self.gen_optimizer.step()
        return {"loss_D": lossD.item(), **G_losses}

    def validation_step(self, real_images: torch.Tensor, target_images: torch.Tensor):
        with torch.no_grad():
            fake_images = self.forward(real_images)
            lossD = self.step_discriminator(real_images, target_images, fake_images)
            _, G_losses = self.step_generator(real_images, target_images, fake_images)
        return {"loss_D": lossD.item(), **G_losses}

    def generate(self, real_images: torch.Tensor, is_scaled: bool = False, to_uint8: bool = False):
        if not is_scaled:
            real_images = real_images.to(dtype=torch.float32)
            real_images = real_images / 255.0
            real_images = (real_images - 0.5) / 0.5
        with torch.no_grad():
            generated_images = self.forward(real_images)
        generated_images = (generated_images + 1) / 2
        if to_uint8:
            generated_images = (generated_images * 255).to(dtype=torch.uint8)
        return generated_images

    def save_model(self, gen_path: str, disc_path: str = None):
        torch.save(self.gen.state_dict(), gen_path)
        if self.is_train and disc_path is not None:
            torch.save(self.disc.state_dict(), disc_path)

    def load_model(self, gen_path: str, disc_path: str = None, device: str = None):
        device = device if device else next(self.gen.parameters()).device
        self.gen.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True), strict=False)
        if disc_path is not None and self.is_train:
            device = device if device else next(self.disc.parameters()).device
            self.disc.load_state_dict(torch.load(gen_path, map_location=device, weights_only=True), strict=False)

    def save_optimizer(self, gen_opt_path: str, disc_opt_path: str = None):
        if self.is_train:
            torch.save(self.gen_optimizer.state_dict(), gen_opt_path)
            if disc_opt_path is not None:
                torch.save(self.disc_optimizer.state_dict(), disc_opt_path)

    def load_optimizer(self, gen_opt_path: str, disc_opt_path: str = None):
        if self.is_train:
            self.gen_optimizer.load_state_dict(torch.load(gen_opt_path, weights_only=True))
            if disc_opt_path is not None:
                self.disc_optimizer.load_state_dict(torch.load(disc_opt_path, weights_only=True))

    def get_current_visuals(self, real_images: torch.Tensor, target_images: torch.Tensor):
        with torch.no_grad():
            fake_images = self.gen(real_images)
        return {"real": real_images, "fake": fake_images, "target": target_images}

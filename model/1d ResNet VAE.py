class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate=0.0, stride=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size_res, padding=1,
                               stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size_res, padding=1,
                               stride=stride)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.downsample_block = None
        
        if self.conv1.in_channels != self.conv2.out_channels:
            self.downsample_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels))

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if out.shape != residual.shape:
            residual = F.interpolate(residual, size=out.shape[-1], mode='nearest')      
        if self.downsample_block is not None:
            residual = self.downsample_block(residual)
        out += residual
        out = self.relu(out)
        return out.to(x.device)
            
class VAE(nn.Module):
    def __init__(self, latent_dim, red, kernel_size, stride, padding, input_dim, kernel_size_res):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.red = red
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_dim = input_dim
        self.kernel_size_res = kernel_size_res
        
        # Encoder layers
        encoder_layers = []
        encoder_layers.append(nn.Conv1d(input_dim, 32* red, kernel_size=kernel_size, stride=stride, padding=padding))
        encoder_layers.append(nn.BatchNorm1d(32* red))
        encoder_layers.append(nn.ReLU(inplace=False)) 
        encoder_layers.append(ResidualBlock(32* red, 32* red, kernel_size=kernel_size, stride=stride))
        
        encoder_layers.append(nn.Conv1d(32* red, 16 * red, kernel_size=kernel_size, stride=stride, padding=padding))
        encoder_layers.append(nn.BatchNorm1d(16 * red))
        encoder_layers.append(nn.ReLU(inplace=False))
        encoder_layers.append(ResidualBlock(16 * red,16 * red, kernel_size=kernel_size, stride=stride))

        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 180, 27) 
            encoder_output_shape, decoder_input_shape = self._get_encoder_output_shape(dummy_input)
            
        self.fc_lin1 = nn.Linear(encoder_output_shape, 1024)
        self.fc_lin2 = nn.Linear(1024, 512)
        self.fc_lin3 = nn.Linear(512, 256)
        self.fc_lin4 = nn.Linear(256, 128)

        self.fc_mu = nn.Linear(128, self.latent_dim)
        self.fc_log_var = nn.Linear(128, self.latent_dim)

        # Decoder layers
        decoder_layers = []
        self.decoder_input = nn.Linear(self.latent_dim, encoder_output_shape)
        self.decoder_reshape = decoder_input_shape

        decoder_layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
        decoder_layers.append(nn.Conv1d(16* red, 32* red, kernel_size=kernel_size, stride=stride, padding=padding)) 
        decoder_layers.append(nn.BatchNorm1d(32* red))
        decoder_layers.append(nn.ReLU(inplace=False))
        decoder_layers.append(ResidualBlock(32* red, 32* red, kernel_size=kernel_size, stride=stride))

        decoder_layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=False))
        decoder_layers.append(nn.Conv1d(32* red, input_dim, kernel_size=kernel_size, stride=stride, padding=padding))      
        decoder_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_layers)
        for layer in encoder_layers:
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                xavier_normal_(layer.weight)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_lin1(x)
        x = self.fc_lin2(x)
        x = self.fc_lin3(x)
        x = self.fc_lin4(x)

        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        batch_size = x.size(0)
        decoder_reshape_without_batch = self.decoder_reshape[1:]
        x = x.view(batch_size, *decoder_reshape_without_batch)
        x_recon = self.decoder(x)
        x_recon = F.interpolate(x_recon, 27, mode='linear', align_corners=False)
        return x_recon

    # Inside the forward method of the VAE class
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, z, mu, log_var 

    def _get_encoder_output_shape(self, x):
        x = self.encoder(x)
        encoder_output_shape = x.view(x.size(0), -1).shape[1]
        decoder_input_shape = x.view(x.size(0), x.size(1), x.size(2)).shape
        return encoder_output_shape, decoder_input_shape

class VAE_Loss(nn.Module):
    def __init__(self, beta):
        super(VAE_Loss, self).__init__()
        self.beta = beta

    def forward(self, decoded, x, mu, log_var):
        reconstruction_loss = F.mse_loss(decoded, x, reduction='sum')
        kl_divergence_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = reconstruction_loss + self.beta * kl_divergence_loss
        return total_loss, reconstruction_loss, kl_divergence_loss
